"use strict";

const RESOURCE_NAMES = ["wood", "brick", "sheep", "wheat", "ore"];
const DEV_NAMES = ["Knight", "Year of Plenty", "Monopoly", "Road Building", "Victory Point"];
const TILE_CLASSES = ["wood", "brick", "sheep", "wheat", "ore", "desert"];
const NS = "http://www.w3.org/2000/svg";

let meta = null;
let game = null;
let botTimer = null;
let requestBusy = false;
let renderedEnd = false;

const $ = (selector) => document.querySelector(selector);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const svg = (tag, attrs = {}) => {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
};

function resourceIcon(name, className = "") {
  const icon = svg("svg", {viewBox: "0 0 32 32", class: `resource-icon ${className}`,
    role: "img", "aria-label": name, focusable: "false"});
  if (name === "wood") {
    icon.append(svg("path", {d: "M13 27h6l-1-8h5l-5-6h4L16 4l-6 9h4l-5 6h5z", class: "icon-fill"}));
  } else if (name === "brick") {
    icon.append(svg("rect", {x: 4, y: 7, width: 11, height: 8, rx: 1, class: "icon-fill"}),
      svg("rect", {x: 17, y: 7, width: 11, height: 8, rx: 1, class: "icon-fill"}),
      svg("rect", {x: 9, y: 17, width: 14, height: 8, rx: 1, class: "icon-fill"}));
  } else if (name === "sheep") {
    icon.append(svg("circle", {cx: 11, cy: 15, r: 6, class: "icon-fill"}),
      svg("circle", {cx: 17, cy: 13, r: 7, class: "icon-fill"}),
      svg("circle", {cx: 23, cy: 16, r: 5, class: "icon-fill"}),
      svg("circle", {cx: 26, cy: 13, r: 3, class: "icon-fill"}),
      svg("path", {d: "M11 20v6m10-6v6", class: "icon-stroke"}));
  } else if (name === "wheat") {
    icon.append(svg("path", {d: "M16 28V6M15 11c-5-1-7-4-7-7 5 1 7 4 7 7zm2 5c5-1 7-4 7-7-5 1-7 4-7 7zm-2 5c-5-1-7-4-7-7 5 1 7 4 7 7zm2 5c5-1 7-4 7-7-5 1-7 4-7 7z", class: "icon-stroke wheat-stroke"}));
  } else if (name === "ore") {
    icon.append(svg("path", {d: "M5 21 10 8l9-4 8 8-3 13-13 3z", class: "icon-fill"}),
      svg("path", {d: "m10 8 6 8 11-4M16 16l-5 12", class: "icon-cut"}));
  }
  return icon;
}

function resourceToken(name, count = 1, {showName = true, compact = false} = {}) {
  const token = el("span", `resource-token ${name} ${compact ? "compact" : ""}`);
  token.setAttribute("aria-label", `${count} ${name}`);
  token.title = `${count} ${name}`;
  token.append(resourceIcon(name));
  token.append(el("b", "resource-count", count));
  if (showName) token.append(el("span", "resource-name", name));
  return token;
}

function resourceChoice(items, separator = " + ") {
  const line = el("span", "choice-resources");
  items.forEach((item, index) => {
    if (index) line.append(document.createTextNode(separator));
    line.append(resourceToken(item.resource, item.count));
  });
  return line;
}

function countedResources(names) {
  return RESOURCE_NAMES.flatMap(resource => {
    const count = names.filter(name => name === resource).length;
    return count ? [{resource, count}] : [];
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function showError(error) {
  const toast = $("#error-toast");
  toast.textContent = error.message || String(error);
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 5000);
}

function seatColor(seat) {
  const color = game.state.players[seat].color;
  return color === 0 ? "#d95a4e" : "#3a78b8";
}

function playerPanel(target, seat, own) {
  const player = game.state.players[seat];
  const panel = $(target);
  panel.style.setProperty("--player-color", seatColor(seat));
  panel.replaceChildren();
  const avatar = el("div", "avatar", own ? "Y" : "B");
  const name = el("div", "player-name", own ? "You" : "Bot");
  const sub = el("small", "", own ? (seat === 0 ? "First seat" : "Second seat") : game.bot.label);
  name.append(sub);
  if (!own) {
    const breakdown = el("div", "player-resource-breakdown");
    player.resources.forEach((count, index) => {
      breakdown.append(resourceToken(RESOURCE_NAMES[index], count, {showName: false, compact: true}));
    });
    name.append(breakdown);
  }
  panel.append(avatar, name);
  const stats = [
    [player.visible_vp, "VP"],
    [player.resource_count, "Cards"],
    [player.dev_count, "Dev"],
    [player.knights_played, "Knights"],
    [player.longest_road, "Road"],
  ];
  stats.forEach(([value, label]) => {
    const stat = el("div", "stat");
    stat.append(el("strong", "", value), el("span", "", label));
    panel.append(stat);
  });
  const badges = el("div", "badges");
  const road = el("div", `badge ${player.has_longest_road ? "on" : ""}`, "↝");
  road.title = `Longest Road (${player.longest_road})`;
  const army = el("div", `badge ${player.has_largest_army ? "on" : ""}`, "♞");
  army.title = "Largest Army";
  badges.append(road, army);
  panel.append(badges);
}

function renderHand() {
  const hand = $("#hand");
  hand.replaceChildren();
  const own = game.state.players[game.human_seat];
  own.resources.forEach((count, index) => {
    const card = el("div", `resource-card ${RESOURCE_NAMES[index]}`);
    const label = el("span");
    label.append(resourceIcon(RESOURCE_NAMES[index]), document.createTextNode(RESOURCE_NAMES[index]));
    card.append(el("strong", "", count), label);
    hand.append(card);
  });
  own.dev_hand.forEach((count, index) => {
    if (!count) return;
    const card = el("div", "dev-card");
    card.append(el("strong", "", count), el("span", "", DEV_NAMES[index]));
    hand.append(card);
  });
}

function boardGeometry() {
  const size = 86;
  const cx = 430;
  const cy = 347;
  const centers = meta.topology.land_coords.map(([q, _s, r]) => ({
    x: cx + Math.sqrt(3) * size * (q + r / 2),
    y: cy + 1.5 * size * r,
  }));
  const sums = Array.from({length: 54}, () => ({x: 0, y: 0, n: 0}));
  centers.forEach((center, tile) => {
    meta.topology.tile_vertices[tile].forEach((vertex, corner) => {
      const angle = (-90 + corner * 60) * Math.PI / 180;
      sums[vertex].x += center.x + size * Math.cos(angle);
      sums[vertex].y += center.y + size * Math.sin(angle);
      sums[vertex].n += 1;
    });
  });
  const vertices = sums.map(point => ({x: point.x / point.n, y: point.y / point.n}));
  return {size, cx, cy, centers, vertices};
}

function polygonPoints(center, size) {
  return Array.from({length: 6}, (_, i) => {
    const angle = (-90 + i * 60) * Math.PI / 180;
    return `${center.x + size * Math.cos(angle)},${center.y + size * Math.sin(angle)}`;
  }).join(" ");
}

function renderBoard() {
  const board = $("#board");
  board.replaceChildren();
  const geo = boardGeometry();
  const legalByVertex = new Map();
  const legalByEdge = new Map();
  const legalByTile = new Map();
  game.legal_actions.forEach(action => {
    if (action.vertex !== undefined) legalByVertex.set(action.vertex, action);
    if (action.edge !== undefined) legalByEdge.set(action.edge, action);
    if (action.tile !== undefined) legalByTile.set(action.tile, action);
  });

  const portLayer = svg("g");
  meta.topology.port_vertices.forEach((pair, index) => {
    const a = geo.vertices[pair[0]], b = geo.vertices[pair[1]];
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    const dx = mx - geo.cx, dy = my - geo.cy;
    const mag = Math.hypot(dx, dy) || 1;
    const px = mx + dx / mag * 62, py = my + dy / mag * 62;
    portLayer.append(svg("line", {x1: a.x, y1: a.y, x2: px, y2: py, class: "port-line"}));
    portLayer.append(svg("line", {x1: b.x, y1: b.y, x2: px, y2: py, class: "port-line"}));
    const type = game.state.port_types[index];
    const portClass = type < 0 ? "port-buoy port-any" : `port-buoy port-${RESOURCE_NAMES[type]}`;
    portLayer.append(svg("circle", {cx: px, cy: py, r: 36, class: portClass}));
    const ratio = svg("text", {x: px, y: type < 0 ? py : py - 9, class: "port-label"});
    ratio.textContent = type < 0 ? "3:1" : "2:1";
    portLayer.append(ratio);
    if (type >= 0) {
      const name = svg("text", {x: px, y: py + 11, class: "port-label port-resource"});
      name.textContent = RESOURCE_NAMES[type];
      portLayer.append(name);
    }
  });
  board.append(portLayer);

  const tileLayer = svg("g");
  geo.centers.forEach((center, tile) => {
    const resource = game.state.tile_resources[tile];
    const points = polygonPoints(center, geo.size - 2);
    tileLayer.append(svg("polygon", {points, class: `tile tile-${TILE_CLASSES[resource]}`}));
    tileLayer.append(svg("polygon", {points: polygonPoints(center, geo.size - 8), class: "tile-shade"}));
    if (legalByTile.has(tile)) {
      const target = svg("polygon", {points, class: "legal-tile", tabindex: "0", role: "button"});
      target.addEventListener("click", () => sendAction(legalByTile.get(tile).id));
      tileLayer.append(target);
    }
    const number = game.state.tile_numbers[tile];
    if (number) {
      tileLayer.append(svg("circle", {cx: center.x, cy: center.y, r: 29, class: "number-token"}));
      const hot = number === 6 || number === 8;
      const text = svg("text", {x: center.x, y: center.y - 4, class: `number-text ${hot ? "hot" : ""}`});
      text.textContent = number;
      tileLayer.append(text);
      const pips = 6 - Math.abs(7 - number);
      const totalWidth = (pips - 1) * 6;
      for (let i = 0; i < pips; i++) {
        tileLayer.append(svg("circle", {cx: center.x - totalWidth / 2 + i * 6, cy: center.y + 17, r: 1.7, class: `pip ${hot ? "hot" : ""}`}));
      }
    }
    if (game.state.robber_tile === tile) {
      const robber = svg("path", {d: `M ${center.x-12} ${center.y+26} Q ${center.x-18} ${center.y-4} ${center.x-8} ${center.y-20} A 10 10 0 1 1 ${center.x+8} ${center.y-20} Q ${center.x+18} ${center.y-4} ${center.x+12} ${center.y+26} Z`, class: "robber"});
      tileLayer.append(robber);
    }
  });
  board.append(tileLayer);

  const roadLayer = svg("g");
  meta.topology.edge_vertices.forEach((pair, edge) => {
    const a = geo.vertices[pair[0]], b = geo.vertices[pair[1]];
    const owner = game.state.edges[edge];
    if (owner >= 0) {
      roadLayer.append(svg("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: "road-outline"}));
      const road = svg("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: "road"});
      road.style.setProperty("--road-color", seatColor(owner));
      roadLayer.append(road);
    }
    if (legalByEdge.has(edge)) {
      const target = svg("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: "legal-edge", tabindex: "0", role: "button"});
      target.addEventListener("click", () => sendAction(legalByEdge.get(edge).id));
      roadLayer.append(target);
    }
  });
  board.append(roadLayer);

  const buildingLayer = svg("g");
  geo.vertices.forEach((point, vertex) => {
    const value = game.state.vertices[vertex];
    if (value >= 0) {
      const owner = value % 2;
      const city = value >= 2;
      const path = city
        ? `M ${point.x-16} ${point.y+13} V ${point.y-5} H ${point.x-6} V ${point.y-17} L ${point.x+5} ${point.y-8} H ${point.x+16} V ${point.y+13} Z`
        : `M ${point.x-13} ${point.y+12} V ${point.y-4} L ${point.x} ${point.y-16} L ${point.x+13} ${point.y-4} V ${point.y+12} Z`;
      const building = svg("path", {d: path, class: "building", fill: seatColor(owner)});
      buildingLayer.append(building);
    }
    if (legalByVertex.has(vertex)) {
      const target = svg("circle", {cx: point.x, cy: point.y, r: 18, class: "legal-vertex", tabindex: "0", role: "button"});
      target.addEventListener("click", () => sendAction(legalByVertex.get(vertex).id));
      buildingLayer.append(target);
    }
  });
  board.append(buildingLayer);
}

function actionButton(label, action, className = "") {
  const button = el("button", `button ${className}`, label);
  button.disabled = requestBusy;
  button.addEventListener("click", () => sendAction(action.id));
  return button;
}

function openChoices(title, description, actions, formatter) {
  const content = $("#choice-content");
  content.replaceChildren(el("h2", "", title), el("p", "", description));
  const grid = el("div", "choice-grid");
  actions.forEach(action => {
    const button = el("button", "choice");
    const formatted = formatter(action);
    button.append(formatted instanceof Node ? formatted : document.createTextNode(formatted));
    button.addEventListener("click", () => { closeChoice(); sendAction(action.id); });
    grid.append(button);
  });
  content.append(grid);
  $("#choice-modal").classList.add("open");
}

function closeChoice() { $("#choice-modal").classList.remove("open"); }

function renderActions() {
  const dock = $("#actions");
  dock.replaceChildren();
  if (!game.human_turn) {
    dock.append(el("span", "action-hint", game.state.winner !== null ? "Game complete" : "Bot is thinking…"));
    return;
  }
  const byKind = Object.groupBy ? Object.groupBy(game.legal_actions, action => action.kind) : game.legal_actions.reduce((out, action) => ((out[action.kind] ||= []).push(action), out), {});
  if (byKind.roll) dock.append(actionButton("Roll dice", byKind.roll[0], "action-roll"));
  if (byKind.end_turn) dock.append(actionButton("End turn", byKind.end_turn[0], "action-end"));
  if (byKind.buy_dev_card) dock.append(actionButton("Buy dev card", byKind.buy_dev_card[0]));
  if (byKind.play_knight) dock.append(actionButton("Play Knight", byKind.play_knight[0], "dev"));
  if (byKind.play_road_building) dock.append(actionButton("Road Building", byKind.play_road_building[0], "dev"));
  if (byKind.maritime_trade) {
    const button = el("button", "button", "Bank / port trade");
    button.addEventListener("click", () => openChoices("Maritime trade", "Only engine-legal bank and port trades are shown.", byKind.maritime_trade,
      action => resourceChoice([{resource: action.give, count: action.rate}, {resource: action.receive, count: 1}], " → ")));
    dock.append(button);
  }
  if (byKind.play_monopoly) {
    const button = el("button", "button dev", "Play Monopoly");
    button.addEventListener("click", () => openChoices("Choose a resource", "Take every matching card from the bot.", byKind.play_monopoly,
      action => resourceChoice([{resource: action.resource, count: game.state.players[game.bot_seat].resources[RESOURCE_NAMES.indexOf(action.resource)]}])));
    dock.append(button);
  }
  if (byKind.play_year_of_plenty) {
    const button = el("button", "button dev", "Year of Plenty");
    button.addEventListener("click", () => openChoices("Choose resources", "Only combinations available in the bank are shown.", byKind.play_year_of_plenty,
      action => resourceChoice(countedResources(action.resources))));
    dock.append(button);
  }
  if (byKind.discard) {
    const box = el("div", "discard-box");
    box.append(el("b", "", `Discard ${game.state.discard_counts[game.human_seat]} more:`));
    byKind.discard.forEach(action => {
      const button = el("button", `resource-chip ${action.resource}`);
      button.append(resourceIcon(action.resource), document.createTextNode(action.resource));
      button.addEventListener("click", () => sendAction(action.id));
      box.append(button);
    });
    dock.replaceChildren(box);
  }
  const spatial = ["build_settlement", "build_road", "build_city", "move_robber"].some(kind => byKind[kind]);
  if (spatial) {
    const prompt = game.state.prompt_name === "robber" ? "Choose a highlighted hex for the robber"
      : game.state.prompt_name === "initial_settlement" ? "Choose a highlighted settlement spot"
      : game.state.prompt_name === "initial_road" ? "Choose a highlighted road"
      : "Highlighted board locations are legal builds";
    dock.append(el("span", "action-hint", prompt));
  }
  if (!dock.childNodes.length) dock.append(el("span", "action-hint", "Choose a highlighted legal action on the board"));
}

function renderLog() {
  const log = $("#game-log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 70;
  log.replaceChildren();
  game.messages.forEach(message => {
    const entry = el("div", `log-entry ${message.kind}`);
    entry.setAttribute("aria-label", message.text);
    (message.parts || [message.text]).forEach(part => {
      entry.append(typeof part === "string"
        ? document.createTextNode(part)
        : resourceToken(part.resource, part.count));
    });
    log.append(entry);
  });
  if (atBottom) log.scrollTop = log.scrollHeight;
  $("#log-file").textContent = `Logging to games/${game.log_file}`;
}

function renderDice() {
  const box = $("#dice");
  if (!game.last_dice) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.replaceChildren(...game.last_dice.map(value => el("div", "die", value)));
}

function renderEnd() {
  if (game.state.winner === null || renderedEnd) return;
  renderedEnd = true;
  const humanWon = game.state.winner === game.human_seat;
  const scores = game.state.players.map(player => player.actual_vp);
  const content = $("#end-content");
  content.replaceChildren();
  content.append(el("div", "result-seal", humanWon ? "♛" : "⚑"));
  content.append(el("h1", "", humanWon ? "Harbor victory!" : "The bot takes it."));
  content.append(el("p", "end-summary", `${scores[game.human_seat]}–${scores[game.bot_seat]} · ${game.state.num_turns} turns · ${formatDuration(game.elapsed_seconds)}`));
  content.append(el("div", "eyebrow", "DICE HISTOGRAM"));
  const hist = el("div", "histogram");
  const max = Math.max(1, ...Object.values(game.dice_histogram));
  for (let total = 2; total <= 12; total++) {
    const col = el("div", "hist-col");
    const bar = el("div", "hist-bar");
    bar.style.height = `${Math.max(3, game.dice_histogram[total] / max * 85)}px`;
    bar.title = `${game.dice_histogram[total]} rolls`;
    col.append(bar, document.createTextNode(total));
    hist.append(col);
  }
  content.append(hist);
  $("#end-modal").classList.add("open");
}

function formatDuration(seconds) {
  const value = Math.round(seconds);
  return `${Math.floor(value / 60)}:${String(value % 60).padStart(2, "0")}`;
}

function statusText() {
  if (game.state.winner !== null) return game.state.winner === game.human_seat ? "You won" : "Bot won";
  if (game.bot_pending) return "Bot is thinking…";
  const prompts = {
    initial_settlement: "Your opening settlement",
    initial_road: "Your opening road",
    discard: `Discard ${game.state.discard_counts[game.human_seat]} cards`,
    robber: "Move the robber",
    play: game.state.has_rolled[game.human_seat] ? "Your move" : "Your turn",
  };
  return prompts[game.state.prompt_name] || "Your move";
}

function render() {
  if (!game) return;
  $("#game-shell").classList.remove("is-empty");
  playerPanel("#opponent-panel", game.bot_seat, false);
  playerPanel("#own-panel", game.human_seat, true);
  renderBoard(); renderHand(); renderActions(); renderLog(); renderDice();
  $("#status-pill").textContent = statusText();
  $("#turn-count").textContent = `Turn ${game.state.num_turns}`;
  renderEnd();
  scheduleBot();
}

function acceptPayload(payload) {
  game = payload;
  render();
}

async function sendAction(action) {
  if (requestBusy) return;
  requestBusy = true;
  renderActions();
  try {
    acceptPayload(await api("/api/action", {method: "POST", body: JSON.stringify({action})}));
  } catch (error) { showError(error); }
  finally { requestBusy = false; if (game) { renderActions(); scheduleBot(); } }
}

function scheduleBot() {
  window.clearTimeout(botTimer);
  if (!game?.bot_pending || game.state.winner !== null || requestBusy) return;
  botTimer = window.setTimeout(async () => {
    requestBusy = true;
    try { acceptPayload(await api("/api/bot-step", {method: "POST", body: "{}"})); }
    catch (error) { showError(error); }
    finally { requestBusy = false; if (game) { renderActions(); scheduleBot(); } }
  }, 400);
}

async function startGame(form) {
  const data = new FormData(form);
  const payload = {seat: data.get("seat"), bot: data.get("bot")};
  if (String(data.get("seed") || "").trim()) payload.seed = Number(data.get("seed"));
  const button = form.querySelector("button[type=submit]");
  button.disabled = true; button.textContent = "Setting the board…";
  try {
    acceptPayload(await api("/api/games", {method: "POST", body: JSON.stringify(payload)}));
    renderedEnd = false;
    $("#setup-modal").classList.remove("open");
    $("#end-modal").classList.remove("open");
  } catch (error) { showError(error); }
  finally { button.disabled = false; button.textContent = "Start game"; }
}

function tickTimer() {
  if (game && $("#timer-toggle").checked && game.state.winner === null) {
    $("#status-pill").textContent = `${statusText()} · ${formatDuration(game.turn_elapsed_seconds)}`;
    game.turn_elapsed_seconds += 1;
    game.elapsed_seconds += 1;
  }
}

async function init() {
  try {
    meta = await api("/api/meta");
    $("#rules").textContent = meta.rules_summary;
    const state = await api("/api/state");
    if (state.active) {
      const {active, ...payload} = state;
      acceptPayload(payload);
      $("#setup-modal").classList.remove("open");
    }
  } catch (error) { showError(error); }
}

$("#setup-form").addEventListener("submit", event => { event.preventDefault(); startGame(event.currentTarget); });
$("#new-game-button").addEventListener("click", () => $("#setup-modal").classList.add("open"));
$("#end-new-game").addEventListener("click", () => { $("#end-modal").classList.remove("open"); $("#setup-modal").classList.add("open"); });
$("#choice-close").addEventListener("click", closeChoice);
window.setInterval(tickTimer, 1000);
init();
