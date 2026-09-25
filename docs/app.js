/* Ledger Sentinel: page logic.
 *
 * Reads window.LEDGER, written by src/export_site.py, and builds the page from it:
 * the headline numbers, the payment-queue explorer, the tables and the charts.
 * Every number on the page comes from that file, so the text can't drift from
 * the results.
 */
(function () {
  "use strict";

  const D = window.LEDGER;
  if (!D) throw new Error("data/site-data.js did not load");

  // ---------------------------------------------------------------- formatting

  const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;
  const int = (v) => v.toLocaleString("en-US");
  const money = (v) => v.toLocaleString("en-US", { style: "currency", currency: "USD" });
  const dollars = (v) => v.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  const signed = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
  const sentence = (s) => `${s.charAt(0).toUpperCase()}${s.slice(1)}.`;

  const DECISIONS = ["ESCALATE", "HOLD", "RELEASE"];
  const LABEL = { ESCALATE: "Escalate", HOLD: "Hold for review", RELEASE: "Release" };
  const SHORT = { ESCALATE: "Escalate", HOLD: "Hold", RELEASE: "Release" };
  const MODEL = {
    tree: "Decision tree (shipped)",
    naive: "Decision tree + region",
    forest: "Random forest",
    logistic: "Logistic regression",
    rule: "Keyword rule (baseline)",
  };

  // Icons are fixed strings written here, never data.
  const ICONS = {
    ESCALATE: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="10" cy="10" r="7.5"/><path d="M4.9 15.1L15.1 4.9"/></svg>',
    HOLD: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="10" cy="10" r="7.5"/><path d="M8 7v6M12 7v6"/></svg>',
    RELEASE: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="10" cy="10" r="7.5"/><path d="M6.6 10.2l2.4 2.4 4.5-5"/></svg>',
    wrong: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="10" cy="10" r="7.5"/><path d="M7.3 7.3l5.4 5.4M12.7 7.3l-5.4 5.4"/></svg>',
  };

  function h(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const key in props || {}) {
      if (key === "class") node.className = props[key];
      else if (key === "text") node.textContent = props[key];
      else node.setAttribute(key, props[key]);
    }
    for (const child of children) if (child != null) node.append(child);
    return node;
  }

  function icon(name) {
    const holder = document.createElement("span");
    holder.innerHTML = ICONS[name];
    return holder.firstChild;
  }

  function meter(risk, decision, className) {
    const fill = h("span", { class: `meter-fill ${decision}` });
    fill.style.width = `${(risk * 100).toFixed(1)}%`;
    return h("span", { class: className || "meter-track" }, fill);
  }

  function table(target, headers, rows) {
    const head = h("tr", {}, ...headers.map((col) => h("th", { class: col.right ? "right" : "", text: col.text })));
    const body = rows.map((row) =>
      h("tr", { class: row.class || "" }, ...row.cells.map((cell) =>
        cell instanceof Node ? h("td", {}, cell) : h("td", { class: cell.right ? "right num" : "", text: cell.text })
      ))
    );
    target.replaceChildren(h("thead", {}, head), h("tbody", {}, ...body));
  }

  // ---------------------------------------------------------------- facts in the prose

  const focus = D.fairness.find((r) => r.region === D.meta.mislabelled_region);
  const others = D.fairness.filter((r) => r.region !== D.meta.mislabelled_region);
  const facts = {
    ledgerInvoices: int(D.meta.ledger_invoices),
    testInvoices: int(D.meta.test_invoices),
    testFraud: int(D.meta.test_fraud),
    escalate: D.meta.escalate_above.toFixed(2),
    release: D.meta.release_below.toFixed(2),
    legitAbove: pct(D.hist.similarity.legit_above),
    rule: pct(D.kpis.rule),
    depth: String(D.meta.tree_depth),
    f1Share: pct(D.uncertainty.f1_gap.share_of_resamples_b_better, 1),
    f1Gap: signed(D.uncertainty.f1_gap.estimate),
    f1GapLo: signed(D.uncertainty.f1_gap.ci_low),
    f1GapHi: signed(D.uncertainty.f1_gap.ci_high),
    reviewed: pct(D.kpis.reviewed),
    dollars: pct(D.kpis.dollars),
    holdFraud: int(D.routing.find((r) => r.action === "HOLD").fraudulent),
    mislabelled: pct(D.meta.mislabelled_share),
    region: D.meta.mislabelled_region,
    naiveRate: pct(focus.naive.rate),
    treeRate: pct(focus.tree.rate),
    treeLo: pct(focus.tree.lo, 1),
    otherHi: pct(Math.max(...others.map((r) => r.tree.hi)), 1),
    costRatio: String(D.meta.cost_ratio),
  };
  document.querySelectorAll("[data-fact]").forEach((node) => {
    const key = node.dataset.fact;
    if (!(key in facts)) throw new Error(`No value for data-fact="${key}"`);
    node.textContent = facts[key];
  });

  // ---------------------------------------------------------------- headline numbers

  const kpis = [
    { label: "Fraud caught", value: D.kpis.caught, sub: "of fraudulent invoices escalated or held" },
    { label: "Sent to review", value: D.kpis.reviewed, sub: "of the payment queue goes to an analyst" },
    { label: "Fraudulent dollars intercepted", value: D.kpis.dollars, sub: "held or blocked before payment" },
    { label: "Keyword rule catches", value: D.kpis.rule, sub: "of fraud: the obvious baseline", baseline: true },
  ];
  document.getElementById("kpis").replaceChildren(
    ...kpis.map((k) =>
      h("div", { class: `kpi${k.baseline ? " is-baseline" : ""}` },
        h("div", { class: "kpi-label", text: k.label }),
        h("div", { class: "kpi-value", text: pct(k.value) }),
        h("div", { class: "kpi-sub", text: k.sub }))
    )
  );

  // ---------------------------------------------------------------- payment queue

  const state = { show: new Set(DECISIONS), query: "", selected: null };
  const caseEl = document.getElementById("case");
  const wrap = document.getElementById("queue-wrap");
  const caption = document.getElementById("queue-caption");
  const narrow = window.matchMedia("(max-width: 1000px)");
  const rows = new Map();

  const body = document.getElementById("queue-body");
  for (const q of D.queue) {
    const tr = h("tr", { tabindex: "0", "aria-selected": "false", "data-id": q.id },
      h("td", { class: "id", text: q.id }),
      h("td", { class: "hide-sm", text: q.vendor }),
      h("td", { class: "right num hide-sm", text: money(q.amount) }),
      h("td", {}, h("span", { class: "meter" }, meter(q.risk, q.decision), h("span", { class: "num", text: q.risk.toFixed(2) }))),
      h("td", {}, h("span", { class: "decision" }, h("span", { class: `dot ${q.decision}` }), SHORT[q.decision])),
      h("td", { class: `hide-sm ${q.fraud ? "truth-fraud" : "truth-clean"}`, text: q.fraud ? "Fraud" : "Clean" })
    );
    tr.addEventListener("click", () => select(q.id, { scrollPage: true }));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select(q.id, { scrollPage: true });
      } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        step(tr, event.key === "ArrowDown" ? 1 : -1);
      }
    });
    rows.set(q.id, { tr, q });
    body.appendChild(tr);
  }

  function step(tr, direction) {
    let next = direction > 0 ? tr.nextElementSibling : tr.previousElementSibling;
    while (next && next.hidden) next = direction > 0 ? next.nextElementSibling : next.previousElementSibling;
    if (next) {
      next.focus();
      select(next.dataset.id, {});
    }
  }

  const counts = {};
  for (const q of D.queue) counts[q.decision] = (counts[q.decision] || 0) + 1;
  const chips = DECISIONS.map((d) => {
    const chip = h("button", { class: "chip", type: "button", "aria-pressed": "true" },
      h("span", { class: `dot ${d}` }), SHORT[d], h("span", { class: "count num", text: int(counts[d] || 0) }));
    chip.addEventListener("click", () => {
      if (state.show.has(d)) state.show.delete(d);
      else state.show.add(d);
      chip.setAttribute("aria-pressed", String(state.show.has(d)));
      applyFilter();
    });
    return chip;
  });
  document.getElementById("chips").replaceChildren(...chips);

  const search = document.getElementById("search");
  search.addEventListener("input", () => {
    state.query = search.value.trim().toLowerCase();
    applyFilter();
  });

  function applyFilter() {
    let visible = 0;
    for (const { tr, q } of rows.values()) {
      const shown = state.show.has(q.decision) &&
        (!state.query || q.id.toLowerCase().includes(state.query) || q.vendor.toLowerCase().includes(state.query));
      tr.hidden = !shown;
      if (shown) visible += 1;
    }
    caption.textContent = visible
      ? `${int(visible)} of ${int(D.queue.length)} invoices shown, highest risk first. Select a row to open it.`
      : "No invoices match these filters.";
  }

  function resetFilters() {
    state.show = new Set(DECISIONS);
    state.query = "";
    search.value = "";
    chips.forEach((chip) => chip.setAttribute("aria-pressed", "true"));
    applyFilter();
  }

  function revealRow(tr) {
    const offset = tr.getBoundingClientRect().top - wrap.getBoundingClientRect().top;
    wrap.scrollTop += offset - wrap.clientHeight / 2 + tr.offsetHeight / 2;
  }

  const showcaseButtons = D.showcase.map((item) => {
    const button = h("button", { class: "btn", type: "button", "aria-pressed": "false", "data-id": item.id, text: item.label });
    button.addEventListener("click", () => {
      resetFilters();
      select(item.id, { reveal: true, scrollPage: true });
    });
    return button;
  });
  document.getElementById("showcase").replaceChildren(...showcaseButtons);

  function select(id, options) {
    if (state.selected) rows.get(state.selected).tr.setAttribute("aria-selected", "false");
    state.selected = id;
    const { tr } = rows.get(id);
    tr.setAttribute("aria-selected", "true");
    showcaseButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.id === id)));
    renderCase(id);
    if (options.reveal) revealRow(tr);
    // On a phone the case sits below the table, so bring it into view.
    if (options.scrollPage && narrow.matches) caseEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderCase(id) {
    const c = D.cases[id];
    const { q } = rows.get(id);

    const steps = h("ol", { class: "steps" }, ...c.steps.map((s) =>
      h("li", {},
        h("div", { class: "step-main" },
          h("b", { text: `${s.label}: ` }),
          s.value,
          s.rule ? h("span", { class: "step-rule", text: ` (rule: ${s.rule})` }) : null),
        h("code", { class: "step-code", text: s.code }))
    ));

    const evidence = h("dl", { class: "evidence" }, ...c.evidence.map((e) =>
      h("div", {}, h("dt", { text: e.label }), h("dd", { text: e.text }))
    ));

    const verdict = h("div", { class: "verdict" },
      h("span", { class: `tag ${c.correct ? "right" : "wrong"}` },
        icon(c.correct ? "RELEASE" : "wrong"), c.correct ? "Agent was right" : "Agent was wrong"),
      h("span", { class: "truth", text: `Ground truth: this invoice is ${c.truth}.` }));

    caseEl.replaceChildren(
      h("div", { class: "case-top" },
        h("span", { class: "case-id", text: id }),
        h("span", { class: "case-amount num", text: money(q.amount) }),
        h("span", { class: "case-vendor", text: `Vendor ${q.vendor}` })),
      h("div", { class: `callout ${c.action}` },
        icon(c.action),
        h("div", {}, h("strong", { text: LABEL[c.action] }), h("span", { text: sentence(c.instruction) }))),
      h("div", { class: "risk" }, "Risk score", meter(c.risk, c.action), h("b", { class: "num", text: c.risk.toFixed(2) })),
      h("h3", { text: "The model's reasoning, in full" }),
      steps,
      h("h3", { text: "The records behind it" }),
      evidence,
      verdict
    );
  }

  applyFilter();
  select(D.showcase[0].id, { reveal: true });

  // ---------------------------------------------------------------- tables

  const swatch = (series, kind) => h("span", { class: `sw-${kind} s-${series}` });
  const withSwatch = (series, kind, label) =>
    h("span", { class: "decision" }, series ? swatch(series, kind) : h("span", { class: "dot", style: "background:var(--ink-2)" }), label);

  table(document.getElementById("models-table"),
    [{ text: "Model" }, { text: "Precision", right: true }, { text: "Recall", right: true }, { text: "F1", right: true },
     { text: "Average precision", right: true }, { text: "Flagged", right: true }, { text: "Missed", right: true }],
    D.models.map((m) => ({
      class: m.key === "tree" ? "shipped" : "",
      cells: [
        withSwatch(m.key === "rule" ? null : m.key, "line", MODEL[m.key]),
        { text: pct(m.precision), right: true },
        { text: pct(m.recall), right: true },
        { text: m.f1.toFixed(2), right: true },
        { text: m.ap === null ? "—" : m.ap.toFixed(2), right: true },
        { text: int(m.flagged), right: true },
        { text: int(m.missed), right: true },
      ],
    })));

  const interval = (triple, format) => `${format(triple[0])} (${format(triple[1])}–${format(triple[2])})`;
  table(document.getElementById("uncertainty-table"),
    [{ text: "95% interval" }, { text: MODEL.tree, right: true }, { text: MODEL.forest, right: true }],
    [["precision", "Precision", (v) => pct(v)], ["recall", "Recall", (v) => pct(v)], ["f1", "F1", (v) => v.toFixed(2)]].map(
      ([key, label, format]) => ({
        cells: [
          { text: label },
          { text: interval(D.uncertainty.tree[key], format), right: true },
          { text: interval(D.uncertainty.forest[key], format), right: true },
        ],
      })
    ));

  const queueTotal = D.routing.reduce((sum, r) => sum + r.invoices, 0);
  table(document.getElementById("routing-table"),
    [{ text: "Decision" }, { text: "Invoices", right: true }, { text: "Share", right: true },
     { text: "Fraudulent", right: true }, { text: "Fraud dollars", right: true }],
    D.routing.map((r) => ({
      cells: [
        h("span", { class: "decision" }, h("span", { class: `dot ${r.action}` }), SHORT[r.action]),
        { text: int(r.invoices), right: true },
        { text: pct(r.invoices / queueTotal), right: true },
        { text: int(r.fraudulent), right: true },
        { text: dollars(r.fraud_dollars), right: true },
      ],
    })));

  const rate = (v) => `${pct(v.rate, 1)} (${pct(v.lo, 1)}–${pct(v.hi, 1)})`;
  table(document.getElementById("fairness-table"),
    [{ text: "Region" }, { text: "Clean invoices", right: true }, { text: MODEL.tree, right: true }, { text: MODEL.naive, right: true }],
    D.fairness.map((r) => ({
      cells: [{ text: r.region }, { text: int(r.clean), right: true }, { text: rate(r.tree), right: true }, { text: rate(r.naive), right: true }],
    })));

  table(document.getElementById("excluded-table"),
    [{ text: "Excluded" }, { text: "Why" }],
    D.excluded.map((e) => ({ cells: [h("code", { class: "id", text: e.name }), { text: sentence(e.reason) }] })));

  table(document.getElementById("features-table"),
    [{ text: "Feature" }, { text: "What it measures" }],
    D.features.map((f) => ({ cells: [{ text: f.label }, { text: sentence(f.meaning) }] })));
  document.getElementById("feature-count").textContent = String(D.features.length);

  // ---------------------------------------------------------------- charts

  const classes = [
    { series: "clean", name: "Legitimate" },
    { series: "fraud", name: "Fraudulent" },
  ];
  Charts.legend(document.getElementById("legend-hist"), classes.map((c) => ({ series: c.series, label: c.name })), "line");
  Charts.histogram(document.getElementById("chart-sim"), {
    edges: D.hist.similarity.edges,
    series: [
      { series: "clean", name: "Legitimate", values: D.hist.similarity.legit },
      { series: "fraud", name: "Fraudulent", values: D.hist.similarity.fraud },
    ],
    xLabel: "Text similarity to the closest earlier invoice",
    tickFormat: (v) => pct(v),
    tipFormat: (v) => pct(v, 1),
    threshold: { x: D.hist.similarity.threshold, label: "keyword rule threshold" },
  });
  Charts.histogram(document.getElementById("chart-cad"), {
    edges: D.hist.cadence.edges,
    series: [
      { series: "clean", name: "Legitimate", values: D.hist.cadence.legit },
      { series: "fraud", name: "Fraudulent", values: D.hist.cadence.fraud },
    ],
    xLabel: "Days since the vendor's last invoice ÷ its usual gap",
    tickFormat: (v) => `${v}×`,
    tipFormat: (v) => `${v.toFixed(2)}×`,
    capLast: true,
  });

  // Line colours follow the validated slot order: tree, region-aware tree, forest, logistic.
  const prSeries = ["tree", "naive", "forest", "logistic"].map((key) => {
    const curve = D.pr.find((p) => p.key === key);
    return { series: key, name: MODEL[key], ap: curve.ap, points: curve.points };
  });
  Charts.legend(document.getElementById("legend-pr"),
    prSeries.map((s) => ({ series: s.series, label: `${s.name} · AP ${s.ap.toFixed(2)}` })), "line");
  Charts.prChart(document.getElementById("chart-pr"), {
    series: prSeries,
    markers: [
      { x: D.operating.rule[0], y: D.operating.rule[1], label: "keyword rule" },
      { x: D.operating.tree[0], y: D.operating.tree[1], label: "shipped operating point", series: "tree", labelAt: [0.74, 0.47] },
    ],
    baseline: { y: D.meta.base_rate, label: `chance level (${pct(D.meta.base_rate)} fraud)` },
  });

  const fairSeries = [
    { key: "tree", series: "tree", name: MODEL.tree },
    { key: "naive", series: "naive", name: MODEL.naive },
  ];
  Charts.legend(document.getElementById("legend-fair"), fairSeries.map((s) => ({ series: s.series, label: s.name })), "box");
  Charts.groupedBars(document.getElementById("chart-fair"), {
    rows: D.fairness.map((r) => ({ label: r.region, clean: r.clean, values: { tree: r.tree, naive: r.naive } })),
    series: fairSeries,
    xMax: Math.ceil(Math.max(...D.fairness.map((r) => r.naive.hi)) * 10) / 10,
    labelRow: D.meta.mislabelled_region,
    xLabel: "Share of clean invoices wrongly flagged",
  });

  const impSeries = [
    { key: "tree", series: "tree", name: MODEL.tree },
    { key: "forest", series: "forest", name: MODEL.forest },
    { key: "logistic", series: "logistic", name: MODEL.logistic },
  ];
  Charts.legend(document.getElementById("legend-imp"), impSeries.map((s) => ({ series: s.series, label: s.name })), "dot");
  Charts.dotPlot(document.getElementById("chart-imp"), {
    rows: D.importance.map((r) => ({ label: r.label, values: { tree: r.tree, forest: r.forest, logistic: r.logistic } })),
    series: impSeries,
    xLabel: "Drop in average precision",
  });

  // ---------------------------------------------------------------- links to the code

  // On GitHub Pages the address is <owner>.github.io/<repo>/, which gives the repo URL.
  // Opened from disk there is no repo to link to, so those links stay hidden.
  if (location.hostname.endsWith(".github.io")) {
    const owner = location.hostname.slice(0, -".github.io".length);
    const repo = location.pathname.split("/").filter(Boolean)[0];
    const base = `https://github.com/${owner}/${repo}`;
    document.querySelectorAll("[data-repo]").forEach((a) => {
      a.href = base + a.dataset.repo;
      a.hidden = false;
    });
    document.querySelectorAll("[data-repo-block]").forEach((block) => {
      block.hidden = false;
    });
  }
})();
