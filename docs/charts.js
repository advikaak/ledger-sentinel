/* Ledger Sentinel: small, dependency-free SVG charts.
 *
 * Every chart re-draws at its container's real width (so text stays readable on a
 * phone), takes its colours from CSS classes (so light and dark themes switch on
 * their own), and has a hover layer. Hover only adds detail: every number is also
 * in a table on the page.
 */
(function () {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const FONT = '12px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

  // ---------------------------------------------------------------- helpers

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, x, y, value, attrs) {
    const node = el("text", Object.assign({ x, y }, attrs || {}), parent);
    node.textContent = value;
    return node;
  }

  function scale(d0, d1, r0, r1) {
    const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
    f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
    return f;
  }

  function niceStep(span, count) {
    const raw = span / Math.max(count, 1);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const err = raw / mag;
    return (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  }

  function ticks(min, max, count) {
    const step = niceStep(max - min, count);
    const out = [];
    for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) {
      out.push(Math.round(v / step) * step);
    }
    return out;
  }

  function niceMax(value, count) {
    const step = niceStep(value, count);
    return Math.ceil(value / step - 1e-9) * step;
  }

  let measurer = null;
  function textWidth(value, font) {
    measurer = measurer || document.createElement("canvas").getContext("2d");
    measurer.font = font || FONT;
    return measurer.measureText(value).width;
  }

  const pct = (digits) => (v) => `${(v * 100).toFixed(digits)}%`;

  // Re-draw whenever the container's width changes.
  function mount(container, draw) {
    let lastWidth = 0;
    const render = () => {
      const width = Math.floor(container.clientWidth);
      if (!width || width === lastWidth) return;
      lastWidth = width;
      container.replaceChildren();
      draw(container, width);
    };
    new ResizeObserver(() => requestAnimationFrame(render)).observe(container);
    render();
  }

  function frame(container, width, height, margin) {
    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img" }, container);
    const g = el("g", { transform: `translate(${margin.left},${margin.top})` }, svg);
    return { svg, g, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  }

  function yGrid(g, y, values, w, format) {
    const axis = el("g", { class: "ax" }, g);
    for (const v of values) {
      el("line", { class: v === 0 ? "baseline" : "gridline", x1: 0, x2: w, y1: y(v), y2: y(v) }, axis);
      text(axis, -8, y(v) + 4, format(v), { "text-anchor": "end" });
    }
  }

  function xGrid(g, x, values, h, format, gridlines) {
    const axis = el("g", { class: "ax" }, g);
    for (const v of values) {
      if (gridlines) el("line", { class: "gridline", x1: x(v), x2: x(v), y1: 0, y2: h }, axis);
      text(axis, x(v), h + 18, format(v), { "text-anchor": "middle" });
    }
  }

  function axisTitle(g, x, y, value, anchor) {
    text(g, x, y, value, { class: "ax-title", "text-anchor": anchor || "middle" });
  }

  // Bar with a 4px rounded data end and a square end at the baseline.
  function barPath(x0, x1, y, h) {
    const w = x1 - x0;
    if (w <= 0) return "";
    const r = Math.min(4, w, h / 2);
    return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
  }

  // ---------------------------------------------------------------- tooltip

  const tip = document.getElementById("tooltip");

  function showTip(event, title, rows) {
    tip.replaceChildren();
    if (title) {
      const t = document.createElement("div");
      t.className = "tip-title";
      t.textContent = title;
      tip.appendChild(t);
    }
    for (const row of rows) {
      const line = document.createElement("div");
      line.className = "tip-row";
      if (row.series) {
        const sw = document.createElement("span");
        sw.className = `sw-dot s-${row.series}`;
        line.appendChild(sw);
      }
      const label = document.createElement("span");
      label.className = "tip-label";
      label.textContent = row.label;
      const value = document.createElement("span");
      value.className = "tip-value";
      value.textContent = row.value;
      line.append(label, value);
      tip.appendChild(line);
    }
    tip.hidden = false;
    moveTip(event);
  }

  function moveTip(event) {
    const gap = 14;
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    let x = event.clientX + gap;
    let y = event.clientY + gap;
    if (x + w > window.innerWidth - 8) x = event.clientX - w - gap;
    if (y + h > window.innerHeight - 8) y = event.clientY - h - gap;
    tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
  }

  function hideTip() {
    tip.hidden = true;
  }

  // ---------------------------------------------------------------- legend

  function legend(container, items, kind) {
    const box = document.createElement("div");
    box.className = "legend";
    for (const item of items) {
      const entry = document.createElement("span");
      const sw = document.createElement("span");
      sw.className = `sw-${kind} s-${item.series}`;
      entry.append(sw, document.createTextNode(item.label));
      box.appendChild(entry);
    }
    container.appendChild(box);
  }

  // ---------------------------------------------------------------- histogram

  // Two classes' distributions as step outlines with a light wash, shares within class.
  function histogram(container, spec) {
    mount(container, (root, width) => {
      const margin = { top: 14, right: 14, bottom: 46, left: 46 };
      const f = frame(root, width, 250, margin);
      const edges = spec.edges;
      const n = edges.length - 1;
      const x = scale(edges[0], edges[n], 0, f.w);
      const top = niceMax(Math.max(...spec.series.flatMap((s) => s.values)), 4);
      const y = scale(0, top, f.h, 0);

      yGrid(f.g, y, ticks(0, top, 4), f.w, pct(0));
      xGrid(f.g, x, ticks(edges[0], edges[n], width < 420 ? 4 : 6), f.h, spec.tickFormat, false);
      axisTitle(f.g, f.w / 2, f.h + 38, spec.xLabel);

      const band = el("rect", { class: "band", y: 0, height: f.h, width: 0, visibility: "hidden" }, f.g);

      for (const s of spec.series) {
        let d = `M${x(edges[0])},${y(0)}`;
        for (let i = 0; i < n; i++) d += `V${y(s.values[i])}H${x(edges[i + 1])}`;
        d += `V${y(0)}`;
        el("path", { class: `wash s-${s.series}`, d: d + "Z" }, f.g);
        el("path", { class: `line s-${s.series}`, d }, f.g);
      }

      if (spec.threshold) {
        const tx = x(spec.threshold.x);
        el("line", { class: "leader", x1: tx, x2: tx, y1: 0, y2: f.h }, f.g);
        text(f.g, tx - 6, 12, spec.threshold.label, { class: "annot", "text-anchor": "end" });
      }

      const hit = el("rect", { class: "hit", x: 0, y: 0, width: f.w, height: f.h }, f.g);
      hit.addEventListener("pointermove", (event) => {
        const px = event.clientX - hit.getBoundingClientRect().left;
        const i = Math.min(n - 1, Math.max(0, Math.floor((px / f.w) * n)));
        band.setAttribute("x", x(edges[i]));
        band.setAttribute("width", x(edges[i + 1]) - x(edges[i]));
        band.setAttribute("visibility", "visible");
        const range =
          i === n - 1 && spec.capLast
            ? `${spec.tipFormat(edges[i])} and above`
            : `${spec.tipFormat(edges[i])} to ${spec.tipFormat(edges[i + 1])}`;
        showTip(
          event,
          range,
          spec.series.map((s) => ({ series: s.series, label: s.name, value: pct(1)(s.values[i]) }))
        );
      });
      hit.addEventListener("pointerleave", () => {
        band.setAttribute("visibility", "hidden");
        hideTip();
      });
    });
  }

  // ---------------------------------------------------------------- precision-recall

  // Precision at a given recall, following the same step drawing as the curve.
  function precisionAt(points, recall) {
    for (let i = 0; i < points.length - 1; i++) {
      if (recall <= points[i][0] && recall > points[i + 1][0]) return points[i][1];
    }
    return recall > points[0][0] ? points[0][1] : points[points.length - 1][1];
  }

  function prChart(container, spec) {
    mount(container, (root, width) => {
      const height = Math.round(Math.min(420, Math.max(300, width * 0.62)));
      const margin = { top: 12, right: 16, bottom: 46, left: 50 };
      const f = frame(root, width, height, margin);
      const x = scale(0, 1.02, 0, f.w);
      const y = scale(0, 1.05, f.h, 0);
      const grid = [0, 0.2, 0.4, 0.6, 0.8, 1];

      yGrid(f.g, y, grid, f.w, pct(0));
      xGrid(f.g, x, grid, f.h, pct(0), true);
      axisTitle(f.g, f.w / 2, f.h + 38, "Recall: share of fraud caught");
      const yTitle = text(f.g, 0, 0, "Precision: share of flags that are fraud", { class: "ax-title", "text-anchor": "middle" });
      yTitle.setAttribute("transform", `translate(${-40},${f.h / 2}) rotate(-90)`);

      el("line", { class: "baseline", x1: 0, x2: f.w, y1: y(spec.baseline.y), y2: y(spec.baseline.y) }, f.g);
      text(f.g, x(0.35), y(spec.baseline.y) - 6, spec.baseline.label, { class: "annot-muted" });

      for (const s of spec.series) {
        const p = s.points;
        let d = `M${x(p[0][0])},${y(p[0][1])}`;
        for (let i = 1; i < p.length; i++) d += `H${x(p[i][0])}V${y(p[i][1])}`;
        el("path", { class: `line s-${s.series}`, d }, f.g);
      }

      for (const m of spec.markers) {
        const cx = x(m.x);
        const cy = y(m.y);
        if (m.labelAt) {
          const lx = x(m.labelAt[0]);
          const ly = y(m.labelAt[1]);
          el("line", { class: "leader", x1: lx + 4, y1: ly - 4, x2: cx - 6, y2: cy + 3 }, f.g);
          text(f.g, lx, ly + 4, m.label, { class: "annot", "text-anchor": "end" });
        } else {
          text(f.g, cx + 10, cy + 4, m.label, { class: "annot" });
        }
        el("circle", { class: m.series ? `dot-mark s-${m.series}` : "ref-dot", cx, cy, r: 5 }, f.g);
      }

      const cross = el("line", { class: "crosshair", y1: 0, y2: f.h, visibility: "hidden" }, f.g);
      const hit = el("rect", { class: "hit", x: 0, y: 0, width: f.w, height: f.h }, f.g);
      hit.addEventListener("pointermove", (event) => {
        const px = event.clientX - hit.getBoundingClientRect().left;
        const recall = Math.min(1, Math.max(0, x.invert(px)));
        cross.setAttribute("x1", x(recall));
        cross.setAttribute("x2", x(recall));
        cross.setAttribute("visibility", "visible");
        showTip(
          event,
          `At ${pct(0)(recall)} recall`,
          spec.series.map((s) => ({ series: s.series, label: s.name, value: pct(0)(precisionAt(s.points, recall)) }))
        );
      });
      hit.addEventListener("pointerleave", () => {
        cross.setAttribute("visibility", "hidden");
        hideTip();
      });
    });
  }

  // ---------------------------------------------------------------- grouped horizontal bars

  function groupedBars(container, spec) {
    mount(container, (root, width) => {
      const labelW = Math.max(...spec.rows.map((r) => textWidth(r.label, '13px system-ui, sans-serif')));
      const margin = { top: 6, right: 48, bottom: 40, left: Math.ceil(labelW) + 18 };
      const thick = 18;
      const gap = 2;
      const rowH = spec.series.length * thick + (spec.series.length - 1) * gap + 22;
      const f = frame(root, width, margin.top + margin.bottom + rowH * spec.rows.length, margin);
      const x = scale(0, spec.xMax, 0, f.w);

      xGrid(f.g, x, ticks(0, spec.xMax, width < 480 ? 4 : 8), f.h, pct(0), true);
      el("line", { class: "baseline", x1: 0, x2: 0, y1: 0, y2: f.h }, f.g);
      axisTitle(f.g, f.w / 2, f.h + 34, spec.xLabel);

      spec.rows.forEach((row, r) => {
        const top = r * rowH + 11;
        const center = top + (rowH - 22) / 2;
        text(f.g, -12, center + 4, row.label, { class: "ax-title", "text-anchor": "end", style: "font-size:13px" });

        spec.series.forEach((s, j) => {
          const v = row.values[s.key];
          const y0 = top + j * (thick + gap);
          el("path", { class: `fill s-${s.series}`, d: barPath(x(0), x(v.rate), y0, thick) }, f.g);
          el("line", { class: "whisker", x1: x(v.lo), x2: x(v.hi), y1: y0 + thick / 2, y2: y0 + thick / 2 }, f.g);
          if (row.label === spec.labelRow) {
            text(f.g, Math.max(x(v.rate), x(v.hi)) + 6, y0 + thick / 2 + 4, pct(0)(v.rate), { class: "value-label" });
          }
          const hit = el("rect", { class: "hit", x: 0, y: y0 - 1, width: f.w, height: thick + 2 }, f.g);
          hit.addEventListener("pointermove", (event) =>
            showTip(event, `${row.label}: ${row.clean} clean invoices`, [
              { series: s.series, label: s.name, value: pct(1)(v.rate) },
              { label: "95% interval", value: `${pct(1)(v.lo)} to ${pct(1)(v.hi)}` },
              { label: "Wrongly flagged", value: `${v.flags} of ${row.clean}` },
            ])
          );
          hit.addEventListener("pointerleave", hideTip);
        });
      });
    });
  }

  // ---------------------------------------------------------------- dot plot

  // One row per feature, one dot per model. On narrow screens each label sits
  // above its row instead of to the left, so nothing is squeezed.
  function dotPlot(container, spec) {
    mount(container, (root, width) => {
      const stacked = width < 560;
      const labelW = Math.max(...spec.rows.map((r) => textWidth(r.label, '13px system-ui, sans-serif')));
      const margin = { top: 6, right: 16, bottom: 44, left: stacked ? 12 : Math.ceil(labelW) + 20 };
      const rowH = stacked ? 52 : 34;
      const f = frame(root, width, margin.top + margin.bottom + rowH * spec.rows.length, margin);

      const values = spec.rows.flatMap((r) => spec.series.map((s) => r.values[s.key]));
      const lo = Math.min(0, ...values);
      const hi = niceMax(Math.max(...values), 5);
      const x = scale(lo - 0.02, hi, 0, f.w);

      xGrid(f.g, x, ticks(0, hi, width < 480 ? 4 : 6), f.h, (v) => v.toFixed(1), true);
      el("line", { class: "baseline", x1: x(0), x2: x(0), y1: 0, y2: f.h }, f.g);
      axisTitle(f.g, f.w / 2, f.h + 38, spec.xLabel);

      const offsets = spec.series.map((_, j) => (j - (spec.series.length - 1) / 2) * (stacked ? 6 : 8));
      spec.rows.forEach((row, r) => {
        const top = r * rowH;
        const center = stacked ? top + 36 : top + rowH / 2;
        if (stacked) {
          text(f.g, 0, top + 16, row.label, { class: "ax-title halo", style: "font-size:13px" });
        } else {
          text(f.g, -12, center + 4, row.label, { class: "ax-title", "text-anchor": "end", style: "font-size:13px" });
        }
        spec.series.forEach((s, j) => {
          const cx = x(row.values[s.key]);
          const cy = center + offsets[j];
          el("circle", { class: `dot-mark s-${s.series}`, cx, cy, r: 4.5 }, f.g);
          const hit = el("circle", { class: "hit", cx, cy, r: 10 }, f.g);
          hit.addEventListener("pointermove", (event) =>
            showTip(event, row.label, [{ series: s.series, label: s.name, value: row.values[s.key].toFixed(3) }])
          );
          hit.addEventListener("pointerleave", hideTip);
        });
      });
    });
  }

  window.Charts = { histogram, prChart, groupedBars, dotPlot, legend };
})();
