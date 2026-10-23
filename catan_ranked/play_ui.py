"""Local single-game HTTP service for playing the fast-engine bot."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any
from urllib.parse import urlparse

from catan_ranked.value_residual_features import TERM_NAMES
from catan_ranked.value_arena import choose_greedy_action
from catan_ranked_fast import Game, decode_action_json, topology_json


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "web" / "play_ui"
DEFAULT_LOG_DIR = ROOT / "games"
WEIGHTS_PATH = ROOT / "weights" / "champion.json"
RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")
DEV_NAMES = ("Knight", "Year of Plenty", "Monopoly", "Road Building", "Victory Point")
BOT_SPECS = {
    "p3": {
        "kind": "weighted_linear",
        "label": "P3 replica-a (best)",
        "max_plans": 128,
        "max_actions": 14,
        "chance_samples": 4,
        "opponent_lookahead": False,
    },
    "champion": {"kind": "champion", "label": "Champion (depth-2 alpha-beta)", "depth": 2},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _decode(action: int) -> dict[str, Any]:
    return json.loads(decode_action_json(int(action)))


def _digest(raw: str) -> str:
    return sha256(raw.encode("utf-8")).hexdigest()


def _dice_faces(total: int, sequence: int) -> list[int]:
    low = max(1, total - 6)
    high = min(6, total - 1)
    first = low + sequence % (high - low + 1)
    return [first, total - first]


def _format_resources(counts: list[int]) -> str:
    parts = []
    for name, count in zip(RESOURCE_NAMES, counts):
        if count > 0:
            parts.append(f"{count} {name}")
    return ", ".join(parts)


def _resource_parts(counts: list[int]) -> list[Any]:
    parts: list[Any] = []
    for name, count in zip(RESOURCE_NAMES, counts):
        if count <= 0:
            continue
        if parts:
            parts.append(", ")
        parts.append({"resource": name, "count": int(count)})
    return parts


def _message(text: str, parts: list[Any] | None = None) -> dict[str, Any]:
    return {"text": text, "parts": parts if parts is not None else [text]}


def replay_log(path: str | Path) -> tuple[Game, dict[str, Any]]:
    """Replay an exact JSONL trajectory using its recorded chance outcomes."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or rows[0].get("event") != "start":
        raise ValueError("log does not start with a start record")
    game = Game(int(rows[0]["seed"]))
    final = None
    for row in rows[1:]:
        if row.get("event") == "action":
            chance = row.get("chance", {})
            game.apply(
                int(row["action_id"]),
                chance.get("roll_total"),
                chance.get("stolen_resource"),
            )
            actual = _digest(game.replay_state_json())
            if actual != row["state_hash_after"]:
                raise ValueError(f"state hash mismatch after action {row['sequence']}")
        elif row.get("event") == "final":
            final = row
    if final is None:
        raise ValueError("log has no final record")
    if _digest(game.replay_state_json()) != final["final_state_hash"]:
        raise ValueError("final state hash mismatch")
    return game, final


class GameSession:
    def __init__(self, seed: int, human_seat: int, bot_kind: str, log_dir: Path):
        if human_seat not in (0, 1):
            raise ValueError("human_seat must be 0 or 1")
        if bot_kind not in BOT_SPECS:
            raise ValueError(f"unknown bot {bot_kind!r}")
        self.seed = int(seed)
        self.human_seat = human_seat
        self.bot_seat = 1 - human_seat
        self.bot_kind = bot_kind
        self.bot_spec = dict(BOT_SPECS[bot_kind])
        self.game = Game(self.seed)
        self.weights, self.weights_hash = self._load_weights()
        self.started_wall = time.time()
        self.started_monotonic = time.monotonic()
        self.turn_started_monotonic = self.started_monotonic
        self.last_turn = json.loads(self.game.state_json(self.human_seat))["current_turn"]
        self.messages: list[dict[str, Any]] = []
        self.dice_histogram: Counter[int] = Counter()
        self.last_dice: list[int] | None = None
        self.sequence = 0
        self.finalized = False
        self.abandoned = False
        self.log_path, self._log = self._open_log(log_dir)
        self._write({
            "schema": "catan-ranked.human-game/1",
            "event": "start",
            "timestamp": _utc_now(),
            "seed": self.seed,
            "human_seat": self.human_seat,
            "bot_seat": self.bot_seat,
            "bot_spec": self.bot_spec,
            "weights_path": str(WEIGHTS_PATH.relative_to(ROOT)) if bot_kind == "p3" else None,
            "weights_sha256": self.weights_hash if bot_kind == "p3" else None,
            "rules": {
                "victory_points": 15,
                "balanced_dice": True,
                "friendly_robber_visible_vp_below": 3,
                "discard_above_cards": 9,
                "player_trades": False,
            },
        })
        self._add_message("Game started. Place your opening settlement." if human_seat == 0
                          else "Game started. Bot has the first placement.", "system")

    @staticmethod
    def _load_weights() -> tuple[tuple[float, ...], str]:
        raw = WEIGHTS_PATH.read_bytes()
        payload = json.loads(raw)
        ordered = tuple(float(payload["weights"][name]) for name in TERM_NAMES)
        if len(ordered) != 24:
            raise ValueError("P3 checkpoint does not contain 24 ordered weights")
        return ordered, sha256(raw).hexdigest()

    def _open_log(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        base = log_dir / f"{stamp}_{self.seed}.jsonl"
        for suffix in range(10_000):
            path = base if suffix == 0 else base.with_name(f"{base.stem}_{suffix}{base.suffix}")
            try:
                return path, path.open("x", encoding="utf-8")
            except FileExistsError:
                continue
        raise RuntimeError("could not allocate a unique game log")

    def _write(self, payload: dict[str, Any]) -> None:
        self._log.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        self._log.flush()

    def _add_message(self, text: str, kind: str = "action",
                     parts: list[Any] | None = None) -> None:
        self.messages.append({"sequence": self.sequence, "text": text, "kind": kind,
                              "parts": parts if parts is not None else [text],
                              "timestamp": _utc_now()})

    def _actor_name(self, seat: int) -> str:
        return "You" if seat == self.human_seat else "Bot"

    def choose_bot_action(self) -> int:
        if self.bot_kind == "champion":
            return int(self.game.champion_action(2))
        search = self.game.turn_search(128, max_actions=14, chance_samples=4,
                                       opponent_lookahead=False)
        values = search.weighted_linear_values(self.weights)
        action_values = search.action_values(values)
        # Current best policy: replica-a + setup draft search/features.
        return int(choose_greedy_action(self.game, action_values,
                                        endgame_dev_card_liquidation=False,
                                        build_spending_guard=False,
                                        setup_resource_coverage=False,
                                        setup_draft_search=True,
                                        setup_draft_features=True,
                                        robber_frontier=False,
                                        weights=self.weights))

    def _action_message(
        self,
        seat: int,
        decoded: dict[str, Any],
        chance: dict[str, Any],
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> list[dict[str, Any]]:
        actor = self._actor_name(seat)
        kind = decoded["kind"]
        if kind == "roll":
            d1, d2 = chance["dice"]
            lines = [_message(f"{actor} rolled {d1}+{d2}")]
            for player in (0, 1):
                delta = [after["resources"][player][i] - before["resources"][player][i]
                         for i in range(5)]
                gained = _format_resources(delta)
                if gained:
                    prefix = f"{self._actor_name(player)} got "
                    lines.append(_message(prefix + gained, [prefix, *_resource_parts(delta)]))
            return lines
        if kind == "build_road":
            return [_message(f"{actor} built a road")]
        if kind == "build_settlement":
            lines = [_message(f"{actor} built a settlement")]
            delta = [after["resources"][seat][i] - before["resources"][seat][i]
                     for i in range(5)]
            gained = _format_resources(delta)
            if gained:
                prefix = f"{actor} got starting resources: "
                lines.append(_message(prefix + gained, [prefix, *_resource_parts(delta)]))
            return lines
        if kind == "build_city":
            return [_message(f"{actor} built a city")]
        if kind == "buy_dev_card":
            if seat == self.human_seat and chance.get("bought_dev") is not None:
                return [_message(f"You bought {DEV_NAMES[chance['bought_dev']]}")]
            return [_message("Bot bought a development card")]
        if kind == "discard":
            resource = decoded["resource"]
            text = f"{actor} discarded 1 {resource}"
            return [_message(text, [f"{actor} discarded ", {"resource": resource, "count": 1}])]
        if kind == "end_turn":
            return [_message(f"{actor} ended the turn")]
        if kind == "maritime_trade":
            text = f"{actor} traded {decoded['rate']} {decoded['give']} for 1 {decoded['receive']}"
            parts = [f"{actor} traded ", {"resource": decoded["give"], "count": decoded["rate"]},
                     " for ", {"resource": decoded["receive"], "count": 1}]
            return [_message(text, parts)]
        if kind == "move_robber":
            stolen = chance.get("stolen_resource_name")
            if stolen:
                text = f"{actor} moved the robber and stole 1 {stolen}"
                return [_message(text, [f"{actor} moved the robber and stole ",
                                        {"resource": stolen, "count": 1}])]
            return [_message(f"{actor} moved the robber")]
        if kind == "play_knight":
            return [_message(f"{actor} played a Knight")]
        if kind == "play_monopoly":
            resource = decoded["resource"]
            resource_index = RESOURCE_NAMES.index(resource)
            amount = int(before["resources"][1 - seat][resource_index])
            text = f"{actor} played Monopoly and took {amount} {resource}"
            return [_message(text, [f"{actor} played Monopoly and took ",
                                    {"resource": resource, "count": amount}])]
        if kind == "play_road_building":
            return [_message(f"{actor} played Road Building")]
        if kind == "play_year_of_plenty":
            counts = [decoded["resources"].count(name) for name in RESOURCE_NAMES]
            detail = _format_resources(counts)
            prefix = f"{actor} played Year of Plenty for "
            return [_message(prefix + detail, [prefix, *_resource_parts(counts)])]
        return [_message(f"{actor} acted")]

    def apply_action(self, actor: str, action: int) -> dict[str, Any]:
        if self.finalized:
            raise ValueError("game is already over")
        expected_seat = self.human_seat if actor == "human" else self.bot_seat
        if self.game.current_player != expected_seat:
            raise ValueError(f"it is not {actor}'s action")
        legal = set(map(int, self.game.legal_actions()))
        if action not in legal:
            raise ValueError(f"illegal action {action}")

        before_raw = self.game.replay_state_json()
        before = json.loads(before_raw)
        decoded = _decode(action)
        roll_total, stolen_resource, bought_dev = self.game.apply_logged(action)
        self.sequence += 1
        chance: dict[str, Any] = {
            "roll_total": roll_total,
            "stolen_resource": stolen_resource,
            "bought_dev": bought_dev,
        }
        if roll_total is not None:
            dice = _dice_faces(roll_total, self.sequence)
            chance["dice"] = dice
            self.last_dice = dice
            self.dice_histogram[roll_total] += 1
        if stolen_resource is not None:
            chance["stolen_resource_name"] = RESOURCE_NAMES[stolen_resource]
        if bought_dev is not None:
            chance["bought_dev_name"] = DEV_NAMES[bought_dev]

        after_raw = self.game.replay_state_json()
        after = json.loads(after_raw)
        for message in self._action_message(expected_seat, decoded, chance, before, after):
            self._add_message(message["text"], parts=message["parts"])
        self._write({
            "schema": "catan-ranked.human-game/1",
            "event": "action",
            "sequence": self.sequence,
            "timestamp": _utc_now(),
            "elapsed_ms": round((time.monotonic() - self.started_monotonic) * 1000),
            "actor": actor,
            "seat": expected_seat,
            "action_id": action,
            "action": decoded,
            "chance": chance,
            "state_hash_before": _digest(before_raw),
            "state_hash_after": _digest(after_raw),
        })
        if after["current_turn"] != self.last_turn:
            self.last_turn = after["current_turn"]
            self.turn_started_monotonic = time.monotonic()
        if self.game.winner >= 0:
            self._finalize()
        return self.public_payload()

    def _finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        final_raw = self.game.replay_state_json()
        scores = list(map(int, self.game.actual_vp))
        winner = int(self.game.winner)
        self._add_message(f"{self._actor_name(winner)} won {scores[winner]}–{scores[1 - winner]}!", "result")
        self._write({
            "schema": "catan-ranked.human-game/1",
            "event": "final",
            "timestamp": _utc_now(),
            "winner": winner,
            "human_won": winner == self.human_seat,
            "scores": scores,
            "turns": int(self.game.num_turns),
            "duration_seconds": round(time.monotonic() - self.started_monotonic, 3),
            "dice_histogram": {str(total): self.dice_histogram[total] for total in range(2, 13)},
            "final_state_hash": _digest(final_raw),
            "final_state": json.loads(final_raw),
        })
        self._log.close()

    def abandon(self) -> None:
        if self.finalized or self.abandoned:
            return
        self.abandoned = True
        self._write({"schema": "catan-ranked.human-game/1", "event": "abandoned",
                     "timestamp": _utc_now(), "sequence": self.sequence})
        self._log.close()

    def public_payload(self) -> dict[str, Any]:
        state = json.loads(self.game.state_json(self.human_seat))
        exact = json.loads(self.game.replay_state_json())
        for seat in (0, 1):
            state["players"][seat]["resources"] = exact["resources"][seat]
        is_human_action = state["winner"] is None and state["current_player"] == self.human_seat
        legal = [_decode(action) for action in self.game.legal_actions()] if is_human_action else []
        elapsed = time.monotonic() - self.started_monotonic
        return {
            "game_id": f"{self.seed}:{int(self.started_wall * 1_000_000)}",
            "state": state,
            "human_seat": self.human_seat,
            "bot_seat": self.bot_seat,
            "bot": self.bot_spec,
            "human_turn": is_human_action,
            "bot_pending": state["winner"] is None and state["current_player"] == self.bot_seat,
            "legal_actions": legal,
            "messages": self.messages,
            "last_dice": self.last_dice,
            "dice_histogram": {str(total): self.dice_histogram[total] for total in range(2, 13)},
            "elapsed_seconds": round(elapsed, 1),
            "turn_elapsed_seconds": round(time.monotonic() - self.turn_started_monotonic, 1),
            "log_file": self.log_path.name,
            "rules_summary": "15 VP · Balanced Dice · Friendly Robber · discard above 9 · bank/port trades only (no player-to-player trades)",
        }


class GameManager:
    def __init__(self, log_dir: Path = DEFAULT_LOG_DIR):
        self.log_dir = Path(log_dir)
        self.session: GameSession | None = None
        self.lock = threading.RLock()

    def new_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        seat_choice = payload.get("seat", "random")
        if seat_choice not in ("first", "second", "random"):
            raise ValueError("seat must be first, second, or random")
        human_seat = {"first": 0, "second": 1}.get(seat_choice, secrets.randbelow(2))
        bot = payload.get("bot", "p3")
        seed = int(payload.get("seed", secrets.randbits(63)))
        if seed < 0 or seed >= 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        with self.lock:
            if self.session is not None:
                self.session.abandon()
            self.session = GameSession(seed, human_seat, bot, self.log_dir)
            return self.session.public_payload()

    def state(self) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                return {"active": False}
            return {"active": True, **self.session.public_payload()}

    def human_action(self, action: int) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise ValueError("no active game")
            return self.session.apply_action("human", int(action))

    def bot_step(self) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise ValueError("no active game")
            if self.session.game.current_player != self.session.bot_seat:
                raise ValueError("the bot does not have an action")
            action = self.session.choose_bot_action()
            return self.session.apply_action("bot", action)


def _handler_class(manager: GameManager, static_dir: Path = STATIC_DIR):
    static_files = {
        "/": (static_dir / "index.html", "text/html; charset=utf-8"),
        "/app.js": (static_dir / "app.js", "text/javascript; charset=utf-8"),
        "/styles.css": (static_dir / "styles.css", "text/css; charset=utf-8"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "CatanPlayUI/1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % args}")

        def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 65_536:
                raise ValueError("request body is too large")
            if length == 0:
                return {}
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/meta":
                self._json({"topology": json.loads(topology_json()), "bots": BOT_SPECS,
                            "rules_summary": "15 VP · Balanced Dice · Friendly Robber · discard above 9 · bank/port trades only (no player-to-player trades)"})
                return
            if path == "/api/state":
                self._json(manager.state())
                return
            static = static_files.get(path)
            if static is None or not static[0].is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            raw = static[0].read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", static[1])
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:  # noqa: N802
            try:
                body = self._body()
                path = urlparse(self.path).path
                if path == "/api/games":
                    result = manager.new_game(body)
                elif path == "/api/action":
                    result = manager.human_action(int(body["action"]))
                elif path == "/api/bot-step":
                    result = manager.bot_step()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._json(result)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # fail visibly without leaking a traceback to the browser
                self._json({"error": f"server error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def create_server(host: str = "127.0.0.1", port: int = 8765,
                  log_dir: Path = DEFAULT_LOG_DIR) -> ThreadingHTTPServer:
    manager = GameManager(log_dir)
    server = ThreadingHTTPServer((host, port), _handler_class(manager))
    server.game_manager = manager
    return server


def main() -> None:
    port = int(os.environ.get("CATAN_PLAY_PORT", "8765"))
    server = create_server(port=port)
    host, port = server.server_address[:2]
    print(f"Play vs Bot is ready at http://{host}:{port}")
    print(f"Game logs: {DEFAULT_LOG_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Play vs Bot.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
