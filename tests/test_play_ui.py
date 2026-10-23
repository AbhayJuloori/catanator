import json
from pathlib import Path
import random
import threading
from urllib.request import Request, urlopen

import pytest

pytest.importorskip("catan_ranked_fast")

from catan_ranked.play_ui import GameSession, create_server, replay_log


def _request(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


@pytest.fixture
def api_server(tmp_path):
    server = create_server(port=0, log_dir=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", tmp_path, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_browser_receives_only_current_engine_legal_actions(api_server):
    base_url, _log_dir, server = api_server
    payload = _request(base_url, "/api/games", {"seat": "first", "bot": "p3", "seed": 73001})
    engine_legal = set(map(int, server.game_manager.session.game.legal_actions()))
    sent = {action["id"] for action in payload["legal_actions"]}
    assert sent == engine_legal
    exact = json.loads(server.game_manager.session.game.replay_state_json())["resources"]
    assert payload["state"]["players"][payload["human_seat"]]["resources"] == exact[payload["human_seat"]]
    assert payload["state"]["players"][payload["bot_seat"]]["resources"] == exact[payload["bot_seat"]]
    assert payload["state"]["players"][payload["bot_seat"]]["dev_hand"] is None


def test_resource_events_have_exact_structured_card_tokens():
    session = object.__new__(GameSession)
    session.human_seat = 0
    before = {"resources": [[1, 1, 1, 1, 1], [2, 3, 4, 5, 6]]}
    after_roll = {"resources": [[2, 1, 1, 3, 1], [2, 4, 4, 5, 7]]}

    roll = session._action_message(0, {"kind": "roll"}, {"dice": [3, 3]}, before, after_roll)
    discard = session._action_message(1, {"kind": "discard", "resource": "brick"}, {}, before, before)
    steal = session._action_message(1, {"kind": "move_robber"},
                                    {"stolen_resource_name": "wheat"}, before, before)
    trade = session._action_message(0, {"kind": "maritime_trade", "rate": 3,
                                        "give": "sheep", "receive": "ore"}, {}, before, before)
    monopoly = session._action_message(0, {"kind": "play_monopoly", "resource": "wood"},
                                       {}, before, before)
    plenty = session._action_message(1, {"kind": "play_year_of_plenty",
                                         "resources": ["wheat", "wheat"]}, {}, before, before)
    hidden_dev = session._action_message(1, {"kind": "buy_dev_card"},
                                          {"bought_dev": 4}, before, before)

    def tokens(message):
        return [part for part in message["parts"] if isinstance(part, dict)]

    assert tokens(roll[1]) == [{"resource": "wood", "count": 1},
                               {"resource": "wheat", "count": 2}]
    assert tokens(roll[2]) == [{"resource": "brick", "count": 1},
                               {"resource": "ore", "count": 1}]
    assert tokens(discard[0]) == [{"resource": "brick", "count": 1}]
    assert tokens(steal[0]) == [{"resource": "wheat", "count": 1}]
    assert tokens(trade[0]) == [{"resource": "sheep", "count": 3},
                                {"resource": "ore", "count": 1}]
    assert tokens(monopoly[0]) == [{"resource": "wood", "count": 2}]
    assert tokens(plenty[0]) == [{"resource": "wheat", "count": 2}]
    assert hidden_dev[0]["text"] == "Bot bought a development card"
    assert tokens(hidden_dev[0]) == []


def test_champion_menu_option_can_take_a_legal_step(api_server):
    base_url, _log_dir, _server = api_server
    payload = _request(base_url, "/api/games", {"seat": "second", "bot": "champion", "seed": 73002})
    assert payload["bot_pending"]
    payload = _request(base_url, "/api/bot-step", {})
    assert payload["messages"][-1]["text"] == "Bot built a settlement"
    assert payload["bot_pending"]


def test_full_game_through_http_and_log_replay(api_server):
    base_url, log_dir, _server = api_server
    rng = random.Random(812377)
    payload = _request(base_url, "/api/games", {"seat": "first", "bot": "p3", "seed": 812377})
    for _ply in range(8_000):
        if payload["state"]["winner"] is not None:
            break
        if payload["bot_pending"]:
            payload = _request(base_url, "/api/bot-step", {})
        else:
            legal = payload["legal_actions"]
            assert legal
            selected = legal[rng.randrange(len(legal))]
            payload = _request(base_url, "/api/action", {"action": selected["id"]})
    else:
        pytest.fail("scripted API game did not finish within 8,000 actions")

    assert payload["state"]["winner"] in (0, 1)
    logs = list(Path(log_dir).glob("*.jsonl"))
    assert len(logs) == 1
    replayed, final = replay_log(logs[0])
    assert replayed.winner == payload["state"]["winner"]
    assert json.loads(replayed.replay_state_json()) == final["final_state"]
    assert final["turns"] == payload["state"]["num_turns"]
    rows = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert rows[0]["event"] == "start"
    assert rows[-1]["event"] == "final"
    assert all("action_id" in row and "chance" in row for row in rows if row["event"] == "action")
