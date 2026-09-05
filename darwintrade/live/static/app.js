"use strict";

const el = (id) => document.getElementById(id);

const state = {
  sessionId: null,
  universe: [],
  running: false,
  maxSymbols: 30,
  report: null,
};

function setStatus(kind, text) {
  el("status-dot").className = `dot ${kind}`;
  el("status-text").textContent = text;
}

function showError(message) {
  const box = el("error");
  box.textContent = message;
  box.hidden = false;
}

function clearError() {
  el("error").hidden = true;
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function money(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function num(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "—";
  const parsed = Number(value);
  return Number.isNaN(parsed) ? "—" : parsed.toFixed(digits);
}

function metric(term, value) {
  const wrap = document.createElement("div");
  const dt = document.createElement("dt");
  dt.textContent = term;
  const dd = document.createElement("dd");
  dd.textContent = value;
  wrap.append(dt, dd);
  return wrap;
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error(`health ${res.status}`);
    const data = await res.json();
    state.universe = data.default_universe || [];
    state.maxSymbols = data.max_symbols || 30;
    if (data.default_capital) el("capital").value = data.default_capital;
    if (data.llm_configured) {
      setStatus("ok", `ready · ${data.data_mode}`);
    } else {
      setStatus("bad", "no LLM key configured");
      showError(
        "No LLM_API_KEY found. Copy .env.example to .env and fill in " +
          "LLM_BASE_URL / LLM_API_KEY / LLM_MODEL, then restart the server."
      );
    }
    el("hint").textContent =
      `Each symbol runs a four-role analyst team · up to ${data.max_symbols} symbols.`;
  } catch (err) {
    setStatus("bad", "server unreachable");
    showError(`Could not reach the API: ${err.message}`);
  }
}

function renderSession(sessionId) {
  const keep = el("keep-memory").checked;
  state.sessionId = keep ? sessionId : null;
  try {
    if (state.sessionId) localStorage.setItem("darwintrade.session", state.sessionId);
    else localStorage.removeItem("darwintrade.session");
  } catch (_) { /* Storage may be disabled by the browser. */ }
  el("capital").disabled = Boolean(state.sessionId);
  const label = el("session-label");
  const reset = el("reset-session");
  if (state.sessionId) {
    label.textContent = `session ${state.sessionId}`;
    label.hidden = false;
    reset.hidden = false;
  } else {
    label.hidden = true;
    reset.hidden = true;
  }
}

function renderSummary(data) {
  el("regime-label").textContent = data.regime.label || "unknown";
  el("regime-conf").textContent = `confidence ${num(data.regime.confidence, 2)}`;

  const exposure = el("exposure");
  exposure.replaceChildren();
  const ex = data.exposure || {};
  exposure.append(
    metric("Gross", pct(ex.gross)),
    metric("Long", pct(ex.long_gross)),
    metric("Short", pct(ex.short_gross)),
    metric("Net", pct(ex.net)),
    metric("Capital", money(data.capital))
  );

  const bits = [
    `Trade date ${data.trade_date}${data.trade_date_inferred ? " (latest)" : ""}`,
    `decision #${data.decision_index}`,
  ];
  if (data.optimizer) bits.push(`allocator ${data.optimizer}`);
  const skipped = Object.keys(data.data?.skipped_symbols || {});
  if (skipped.length) bits.push(`skipped ${skipped.join(", ")}`);
  el("meta").textContent = bits.join(" · ");
}

function renderPositions(data) {
  const body = el("positions");
  body.replaceChildren();
  const rows = data.positions || [];
  const empty = el("no-positions");

  if (!rows.length) {
    empty.hidden = false;
    empty.textContent =
      data.no_trade_reason
        ? `No positions: ${data.no_trade_reason}.`
        : "No position cleared the confidence threshold on this bar.";
    el("positions-table").hidden = true;
    return;
  }
  empty.hidden = true;
  el("positions-table").hidden = false;

  for (const row of rows) {
    const tr = document.createElement("tr");

    const symbol = document.createElement("th");
    symbol.scope = "row";
    symbol.textContent = row.symbol;

    const side = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `side ${row.direction}`;
    badge.textContent = row.direction;
    side.append(badge);

    const weight = document.createElement("td");
    weight.className = "num";
    weight.textContent = pct(row.target_weight);

    const notional = document.createElement("td");
    notional.className = "num";
    notional.textContent = money(row.target_notional);

    const price = document.createElement("td");
    price.className = "num";
    price.textContent = num(row.reference_price, 2);

    const conf = document.createElement("td");
    conf.className = "num";
    conf.textContent = num(row.confidence, 2);

    const thesis = document.createElement("td");
    thesis.className = "thesis";
    thesis.textContent = row.thesis || "—";
    if (row.risk_flags && row.risk_flags.length) {
      const flags = document.createElement("div");
      flags.className = "flags";
      flags.textContent = `flags: ${row.risk_flags.join(", ")}`;
      thesis.append(flags);
    }

    tr.append(symbol, side, weight, notional, price, conf, thesis);
    body.append(tr);
  }
}

function renderMemory(data) {
  const memory = data.memory || {};
  const box = el("memory");
  box.replaceChildren();
  box.append(
    metric("Layer", memory.source_layer || "—"),
    metric("Episodes", memory.total_episodes ?? 0),
    metric("Haircut", num(memory.position_haircut, 2)),
    metric("Outcome fed back", memory.outcome_released ? "yes" : "not yet"),
    metric("Avoided", (memory.avoid_symbols || []).join(", ") || "none")
  );
  if (memory.rationale) box.append(metric("Rationale", memory.rationale));
  if (memory.strategic_patch) {
    box.append(
      metric("Strategic patch", JSON.stringify(memory.strategic_patch.patch || {}))
    );
  }

  const capsules = (memory.capsules || []).filter((c) => (c.n ?? c.samples ?? 0) > 0);
  const wrap = el("capsule-wrap");
  const body = el("capsules");
  body.replaceChildren();
  if (!capsules.length) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  for (const cap of capsules) {
    const tr = document.createElement("tr");
    for (const [value, isNum] of [
      [cap.regime, false],
      [cap.role, false],
      [num(cap.ic, 3), true],
      [cap.n ?? cap.samples ?? 0, true],
    ]) {
      const td = document.createElement("td");
      if (isNum) td.className = "num";
      td.textContent = value;
      tr.append(td);
    }
    body.append(tr);
  }
}

function renderWatchlist(data) {
  const rows = data.watchlist || [];
  const panel = el("watchlist-panel");
  const body = el("watchlist");
  body.replaceChildren();
  if (!rows.length) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  for (const row of rows) {
    const tr = document.createElement("tr");
    const symbol = document.createElement("th");
    symbol.scope = "row";
    symbol.textContent = row.symbol;
    const dir = document.createElement("td");
    dir.textContent = row.direction;
    const conf = document.createElement("td");
    conf.className = "num";
    conf.textContent = num(row.confidence, 2);
    const why = document.createElement("td");
    why.className = "thesis";
    why.textContent = [row.reason, row.thesis].filter(Boolean).join(" · ") || "—";
    tr.append(symbol, dir, conf, why);
    body.append(tr);
  }
}

function render(data) {
  state.report = data;
  el("report-title").textContent = data.sample ? "Sample report · illustrative data" : "Your portfolio view";
  el("empty-state").hidden = true;
  el("data-notes").textContent = [
    ...(data.data?.warnings || []),
    ...Object.entries(data.data?.skipped_symbols || {}).map(([symbol, reason]) => `${symbol}: ${reason}`),
  ].join(" · ");
  renderSummary(data);
  renderPositions(data);
  renderMemory(data);
  renderWatchlist(data);
  el("disclaimer").textContent = data.disclaimer || "";
  el("results").hidden = false;
  if (!data.sample) renderSession(data.session_id);
}

async function run(event) {
  event.preventDefault();
  if (state.running) return;

  const symbols = el("symbols").value.trim();
  if (!symbols) {
    showError("Enter at least one symbol.");
    return;
  }
  const normalized = [...new Set(symbols.toUpperCase().split(/[\s,]+/).filter(Boolean))];
  if (normalized.length > state.maxSymbols) {
    showError(`Use at most ${state.maxSymbols} symbols per analysis.`);
    return;
  }

  state.running = true;
  clearError();
  el("progress").hidden = false;
  el("run").disabled = true;
  el("results").hidden = true;
  el("empty-state").hidden = true;
  el("form").setAttribute("aria-busy", "true");
  const started = Date.now();
  const timer = setInterval(() => {
    el("progress-title").textContent = `Analysing ${normalized.length} symbols · ${Math.floor((Date.now() - started) / 1000)}s elapsed`;
  }, 1000);
  setStatus("warn", "running…");

  const body = {
    symbols: normalized,
    trade_date: el("date").value || undefined,
  };
  if (el("keep-memory").checked && state.sessionId) {
    body.session_id = state.sessionId;
  } else {
    body.capital = Number(el("capital").value);
  }

  try {
    const res = await fetch("/api/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(payload.detail)
        ? payload.detail.map((item) => `${item.loc?.slice(1).join(".") || "Input"}: ${item.msg}`).join("; ")
        : payload.detail;
      throw new Error(detail || `Request failed (${res.status}). Check the server and try again.`);
    }
    render(payload);
    setStatus("ok", "done");
    el("report-title").focus({ preventScroll: true });
    el("results").scrollIntoView({ behavior: "auto", block: "start" });
  } catch (err) {
    showError(err.message);
    setStatus("bad", "failed");
    el("empty-state").hidden = false;
  } finally {
    clearInterval(timer);
    el("form").removeAttribute("aria-busy");
    state.running = false;
    el("progress").hidden = true;
    el("run").disabled = false;
  }
}

el("form").addEventListener("submit", run);

el("use-universe").addEventListener("click", () => {
  if (state.universe.length) el("symbols").value = state.universe.join(", ");
});

el("reset-session").addEventListener("click", () => {
  state.sessionId = null;
  renderSession(null);
});

el("keep-memory").addEventListener("change", (event) => {
  if (!event.target.checked) {
    state.sessionId = null;
    renderSession(null);
  }
});

document.querySelectorAll("[data-symbols]").forEach((button) => {
  button.addEventListener("click", () => { el("symbols").value = button.dataset.symbols; });
});

el("download").addEventListener("click", () => {
  if (!state.report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.report, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `darwintrade-${state.report.trade_date}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

el("example").addEventListener("click", async () => {
  if (state.running) return;
  try {
    const response = await fetch("/static/sample-report.json");
    if (!response.ok) throw new Error("Sample report could not be loaded.");
    render(await response.json());
    el("results").scrollIntoView({ block: "start" });
  } catch (error) { showError(error.message); }
});

async function initialize() {
  await loadHealth();
  try {
    const saved = localStorage.getItem("darwintrade.session");
    if (!saved) return;
    const response = await fetch(`/api/sessions/${encodeURIComponent(saved)}`);
    if (response.ok) renderSession(saved);
    else if (response.status === 404) localStorage.removeItem("darwintrade.session");
  } catch (_) { /* A temporarily unreachable server must not discard saved memory. */ }
}
initialize();
