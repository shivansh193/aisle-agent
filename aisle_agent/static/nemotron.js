/* NVIDIA Nemotron on Nebius Token Factory: engine badge, model-trace drawer, shop assistant,
   and the exit advisor's block. Shared by the console and the exit check. */
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const key = () => { const k = $("key"); if (k && k.value) return k.value;
    try { return JSON.parse(localStorage.getItem("ss_key") || '""'); } catch (e) { return ""; } };
  const hdr = (x = {}) => Object.assign(key() ? {Authorization: `Bearer ${key()}`} : {}, x);
  const short = (m) => String(m || "").replace("nvidia/", "").replace("NVIDIA-", "");
  const money = (v) => "$" + (v || 0).toFixed(v < 0.01 ? 5 : 3);
  let engineLabel = null;

  // ---- badge + trace drawer
  async function badge() {
    const top = document.querySelector(".topright");
    if (!top) return;
    try {
      const h = await (await fetch("/health")).json();
      engineLabel = h.engine_label;
    } catch (e) { return; }
    const b = document.createElement("button");
    b.className = "engbadge"; b.type = "button";
    b.title = engineLabel ? `${engineLabel}: open the model trace` : "Model trace";
    b.innerHTML = engineLabel ? `<span class="nv" aria-hidden="true"></span>Nemotron · Token Factory` : "Model trace";
    b.onclick = toggle;
    top.prepend(b);
    const d = document.createElement("aside");
    d.id = "trace"; d.className = "tracedrawer"; d.hidden = true; d.setAttribute("aria-label", "Model trace");
    d.innerHTML = `<div class="row"><h2 style="font-size:16px">Model trace</h2><button class="x" id="traceX" aria-label="Close">×</button></div>
      <p class="muted" id="traceEng" style="margin:4px 0 10px"></p><div class="tracetot mono" id="traceTot"></div><ol id="traceList"></ol>`;
    document.body.appendChild(d);
    $("traceX").onclick = toggle;
  }
  let timer = null;
  function toggle() {
    const d = $("trace"); d.hidden = !d.hidden;
    clearInterval(timer);
    if (!d.hidden) { load(); timer = setInterval(load, 2500); }
  }
  async function load() {
    try {
      const r = await fetch("/v1/models/trace", {headers: hdr()});
      if (!r.ok) { $("traceEng").textContent = r.status === 401 ? "Enter the shop key." : `trace: ${r.status}`; return; }
      const t = await r.json();
      $("traceEng").textContent = `Engine: ${t.engine}. Pose and photo matching run locally on every frame; Nemotron runs only when a hand visits a shelf.`;
      const calls = t.sources.flatMap((s) => s.calls || []).sort((a, b) => b.t - a.t).slice(0, 60);
      const u = t.sources.reduce((a, s) => ({calls: a.calls + (s.usage?.calls || 0), cost: a.cost + (s.usage?.cost_usd || 0),
        tin: a.tin + (s.usage?.prompt_tokens || 0), tout: a.tout + (s.usage?.completion_tokens || 0), err: a.err + (s.usage?.errors || 0)}),
        {calls: 0, cost: 0, tin: 0, tout: 0, err: 0});
      $("traceTot").innerHTML = `<span><b>${u.calls}</b> calls</span><span><b>${(u.tin + u.tout).toLocaleString()}</b> tokens</span><span><b>${money(u.cost)}</b></span>${u.err ? `<span class="bad"><b>${u.err}</b> failed</span>` : ""}`;
      $("traceList").innerHTML = calls.map((c) => `<li><div class="row"><b>${esc(short(c.model))}</b><span class="mono muted">${new Date(c.t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"})}</span></div>
        <div class="mono muted">${esc(c.purpose || "")}${c.error ? ` · <span class="bad">${esc(c.error)}</span>` : ` · ${c.ms} ms · ${(c.prompt_tokens || 0) + (c.completion_tokens || 0)} tok · ${money(c.cost_usd)}`}</div></li>`).join("")
        || '<li class="muted">No hosted calls yet. They appear when a hand visits a shelf zone.</li>';
    } catch (e) {}
  }

  // ---- shop assistant (console)
  function assistant() {
    const mount = $("assistantMount");
    if (!mount) return;
    mount.innerHTML = `<section class="card white" aria-labelledby="as-h">
      <div class="row"><h2 id="as-h">Ask the shop</h2><span class="mono muted" id="asModel">Nemotron</span></div>
      <div class="chips" id="asChips"></div>
      <ol class="chat" id="asLog" aria-live="polite"></ol>
      <form id="asForm" class="asform"><label class="sr" for="asQ">Question</label>
        <input class="field" id="asQ" placeholder="What's running low?" autocomplete="off"><button class="btn ink" id="asGo">Ask</button></form>
    </section>`;
    const ex = ["What's running low?", "Which shelf had the most uncertain takes, and why?", "Any price labels wrong?"];
    $("asChips").innerHTML = ex.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
    $("asChips").onclick = (e) => { const b = e.target.closest("button"); if (b) { $("asQ").value = b.textContent; ask(); } };
    $("asForm").onsubmit = (e) => { e.preventDefault(); ask(); };
  }
  const history = [];
  async function ask() {
    const q = $("asQ").value.trim(); if (!q) return;
    $("asQ").value = ""; $("asGo").disabled = true;
    const li = document.createElement("li"); li.className = "me"; li.textContent = q; $("asLog").appendChild(li);
    const a = document.createElement("li"); a.className = "bot"; a.innerHTML = '<span class="muted">thinking…</span>'; $("asLog").appendChild(a);
    try {
      const r = await fetch("/v1/assistant", {method: "POST", headers: hdr({"Content-Type": "application/json"}), body: JSON.stringify({question: q, history})});
      const out = await r.json();
      if (!r.ok) throw new Error(out.detail || r.status);
      a.innerHTML = `${esc(out.answer)}<div class="mono muted meta">${esc(short(out.model))}${out.tools && out.tools.length ? " · used " + esc([...new Set(out.tools)].join(", ")) : ""}</div>`;
      $("asModel").textContent = short(out.model);
      history.push({role: "user", content: q}, {role: "assistant", content: out.answer});
    } catch (e) { a.innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
    $("asGo").disabled = false; a.scrollIntoView({block: "nearest"});
  }

  // ---- exit advisor
  const REC = {keep: ["Keep on bill", "var(--green)"], remove: ["Remove", "var(--red)"], ask_shopper: ["Ask the shopper", "var(--amber)"]};
  window.loadAdvice = async (sid, id) => {
    const box = $("advice"); if (!box) return;
    box.innerHTML = '<div class="advice"><span class="muted">Nemotron-3-Super is reading the evidence…</span></div>';
    try {
      const r = await fetch(`/v1/sessions/${sid}/bill/${id}/advice`, {headers: hdr()});
      const a = await r.json();
      if (!r.ok) throw new Error(a.detail || r.status);
      const [label, color] = REC[a.recommendation] || REC.ask_shopper;
      const caps = Object.entries(a.captions || {}).map(([k, v]) => `<li><span class="mono muted">${esc(k.replace("_", " · "))}</span> ${esc(v)}</li>`).join("");
      box.innerHTML = `<div class="advice"><div class="row"><span class="h3" style="margin:0">Advisor · ${esc(short(a.model))}</span>
        <span class="mono" style="color:${color};font-weight:600">${label} · ${Number(a.confidence).toFixed(2)}</span></div>
        <p style="margin:8px 0 0">${esc(a.explanation)}</p>${caps ? `<ul class="caps">${caps}</ul>` : ""}
        <p class="muted" style="margin:8px 0 0;font-size:12.5px">Advice only. You decide; both go in the record.</p></div>`;
    } catch (e) { box.innerHTML = `<div class="advice muted">Advisor unavailable (${esc(e.message)}). Decide from the evidence above.</div>`; }
    agreement();
  };
  async function agreement() {
    const el = $("agentStats"); if (!el) return;
    try {
      const s = await (await fetch("/v1/agents/stats", {headers: hdr()})).json();
      el.innerHTML = s.with_advice ? `Advisor agreed with staff on <b>${s.agreed} of ${s.with_advice}</b> decisions (${Math.round(s.agreement_rate * 100)}%)`
        : s.decisions ? `${s.decisions} decision${s.decisions > 1 ? "s" : ""} so far; the advisor said "ask the shopper" on each, so there's nothing to compare yet.` : "Advisor agreement appears after the first decision.";
    } catch (e) {}
  }
  window.refreshAgreement = agreement;

  // ---- end-of-day report (layout page)
  function dayReport() {
    const btn = $("dayReport"), mount = $("reportMount");
    if (!btn || !mount) return;
    btn.onclick = async () => {
      btn.disabled = true;
      mount.innerHTML = '<section class="card white"><h2>End-of-day report</h2><p class="muted">Nemotron-3-Ultra is reading the day&rsquo;s bills, events, stock and exit decisions…</p></section>';
      try {
        const r = await fetch("/v1/report", {method: "POST", headers: hdr()});
        const out = await r.json();
        if (!r.ok) throw new Error(out.detail || r.status);
        const d = out.report;
        const list = (xs, f) => (xs && xs.length) ? `<ul class="rep">${xs.map(f).join("")}</ul>` : '<p class="muted">None.</p>';
        mount.innerHTML = `<section class="card white" aria-labelledby="rep-h"><div class="row"><h2 id="rep-h">End-of-day report</h2><span class="mono muted">${esc(short(out.model))}</span></div>
          ${d ? `<p style="margin:8px 0 0;font-weight:600">${esc(d.headline)}</p>
          <div class="h3r">Sales by shelf</div>${list(d.sales_by_shelf, (x) => `<li><b>${esc(x.shelf)}</b> · ${esc(x.items)} items · ₹${esc(x.revenue)}</li>`)}
          <div class="h3r">Restock</div>${list(d.restock, (x) => `<li><b>${esc(x.shelf)}</b>: ${esc(x.product)} (${esc(x.left)} left) <span class="muted">${esc(x.why)}</span></li>`)}
          <div class="h3r">Uncertain takes</div>${list(d.uncertain_hotspots, (x) => `<li><b>${esc(x.shelf)}</b> · ${esc(x.count)}: ${esc(x.likely_cause)} <span class="muted">→ ${esc(x.fix)}</span></li>`)}
          <div class="h3r">Exit check</div><p style="margin:4px 0 0">${esc(d.exit_check)}</p>
          <div class="h3r">Tomorrow</div>${list(d.tomorrow, (x) => `<li>${esc(x)}</li>`)}` : `<pre style="white-space:pre-wrap">${esc(out.raw)}</pre>`}
        </section>`;
      } catch (e) { mount.innerHTML = `<section class="card"><p class="bad">Report unavailable: ${esc(e.message)}</p></section>`; }
      btn.disabled = false;
    };
  }

  // ---- shelf audit (console and layout): wrong-shelf products, price labels
  const TAG = {misplaced: ["wrong shelf", "bad"], price: ["price label", "warn"]};
  const ACTIONS = {misplaced: [["moved", "Moved it back", true], ["belongs", "It belongs here", false]],
                   price: [["label_fixed", "Label fixed", true], ["use_label_price", "Charge the label price", false]]};
  const capital = (x) => { x = String(x ?? ""); return x.charAt(0).toUpperCase() + x.slice(1); };
  function findingText(f) {
    if (f.kind === "misplaced") return `${esc(capital(f.item))}, not on this shelf&rsquo;s list${f.expected && f.expected.length ? ` (${esc(f.expected.join(", "))})` : ""}.`;
    return `The label reads <b class="mono">₹${esc(f.label_price)}</b>; the bill charges <b class="mono">₹${esc(f.catalogue_price)}</b> for ${esc(f.product)}.`;
  }
  function findingMeta(f) {
    return f.kind === "misplaced" ? `P(not stocked here) ${Number(f.p).toFixed(2)} · seen ${f.seen > 1 ? f.seen + " times in a row" : "once"}`
      : `read from the label: &ldquo;${esc(f.label_text || "")}&rdquo;`;
  }
  const INTRO = "While no hand is on a shelf, the camera checks that each shelf holds only its own products and that its price labels match the catalogue.";
  async function loadAudit() {
    const mount = $("auditMount"); if (!mount) return;
    let data;
    try { const r = await fetch("/v1/audit", {headers: hdr()}); if (!r.ok) return; data = await r.json(); } catch (e) { return; }
    const chip = $("auditChip");
    if (chip) { chip.hidden = !data.open; chip.textContent = `audit ${data.open} finding${data.open === 1 ? "" : "s"}`; }
    if (!data.audits.length) {
      mount.innerHTML = `<section class="card white audit" aria-labelledby="au-h"><div class="row"><h2 id="au-h">Shelf audit</h2><span class="mono muted">Nemotron-3.5-Super-VL</span></div>
        <p class="muted" style="margin:6px 0 0">${INTRO} Press &ldquo;Audit shelves&rdquo; to run it now.</p></section>`;
      return;
    }
    const last = data.audits[0];
    const rows = data.audits.flatMap((a) => a.findings.map((f) => ({a, f})));
    const okZones = data.audits.filter((a) => !a.findings.length && !a.error);
    mount.innerHTML = `<section class="card white audit" aria-labelledby="au-h">
      <div class="row"><h2 id="au-h">Shelf audit</h2><span class="mono muted">${esc(short(last.model || ""))} · ${data.open} open</span></div>
      <p class="muted" style="margin:6px 0 4px">${INTRO}</p>
      ${rows.map(({a, f}) => `<div class="finding"><div class="thumb">${a.crop ? `<img src="${a.crop}" alt="${esc(a.zone)} shelf when audited">` : ""}</div>
        <div class="fbody"><div class="ftop"><span class="chip ${TAG[f.kind][1]}">${TAG[f.kind][0]}</span><b>${esc(a.zone)}</b></div>
          <div class="ftext">${findingText(f)}</div><div class="mono muted fmeta">${findingMeta(f)}</div></div>
        <div class="facts">${ACTIONS[f.kind].map(([act, label, main]) => `<button type="button" class="btn ${main ? "" : "plain"}" data-s="${esc(a.session)}" data-z="${esc(a.zone)}" data-i="${f.idx}" data-a="${act}">${label}</button>`).join("")}</div></div>`).join("")}
      ${okZones.map((a) => `<div class="fok"><span><b>${esc(a.zone)}</b> · all products on their shelf${a.labels && a.labels.length ? ` · ${a.labels.length} label${a.labels.length > 1 ? "s" : ""} match` : ""}</span><span class="mono muted">ok</span></div>`).join("")}
      ${data.audits.filter((a) => a.error).map((a) => `<div class="fok"><span><b>${esc(a.zone)}</b> · not checked: model unavailable</span><span class="mono muted">retry</span></div>`).join("")}
    </section>`;
    mount.querySelectorAll("button[data-a]").forEach((b) => b.onclick = async () => {
      b.disabled = true;
      await fetch("/v1/audit/resolve", {method: "POST", headers: hdr({"Content-Type": "application/json"}),
        body: JSON.stringify({session: b.dataset.s, zone: b.dataset.z, idx: Number(b.dataset.i), action: b.dataset.a})});
      loadAudit(); if (window.onAudit) window.onAudit();
    });
  }
  window.loadAudit = loadAudit;
  function auditButton() {
    const b = $("auditBtn"); if (!b) return;
    b.onclick = async () => {
      b.disabled = true; const label = b.textContent; b.textContent = "Auditing…";
      const sid = (typeof session !== "undefined" && session) ? session.session_id : null;
      try { await fetch(sid ? `/v1/sessions/${sid}/audit` : "/v1/audit/run", {method: "POST", headers: hdr()}); } catch (e) {}
      for (let i = 0; i < 12; i++) { await new Promise((r) => setTimeout(r, 1500)); await loadAudit(); if (window.onAudit) window.onAudit(); }
      b.textContent = label; b.disabled = false;
    };
  }
  if ($("auditMount")) { loadAudit(); setInterval(loadAudit, 6000); }

  badge(); assistant(); agreement(); dayReport(); auditButton();
})();
