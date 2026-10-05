(() => {
  const $ = (id) => document.getElementById(id);
  const tooltip = $("vizTooltip");

  // Sequential single-hue ramp (light -> dark) for the confusion heat map.
  const SEQ_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
  ];
  const ZERO_CELL = "#fcfcfb";
  const DIAGONAL_CELL = "#e1e0d9";

  const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;
  const fmt = (n) => (n == null ? "—" : n.toLocaleString());

  function el(tag, attrs = {}, text) {
    const node = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text != null) node.textContent = text;
    return node;
  }

  /* ---------- tooltip ---------- */

  function showTip(html, x, y) {
    tooltip.innerHTML = html;
    tooltip.classList.remove("hidden");
    const pad = 14;
    const rect = tooltip.getBoundingClientRect();
    let left = x + pad;
    let top = y + pad;
    if (left + rect.width > window.innerWidth - 8) left = x - rect.width - pad;
    if (top + rect.height > window.innerHeight - 8) top = y - rect.height - pad;
    tooltip.style.left = `${Math.max(8, left)}px`;
    tooltip.style.top = `${Math.max(8, top)}px`;
  }

  function hideTip() {
    tooltip.classList.add("hidden");
  }

  function bindTip(node, html) {
    node.addEventListener("mousemove", (e) => showTip(html(), e.clientX, e.clientY));
    node.addEventListener("mouseleave", hideTip);
    node.addEventListener("focus", () => {
      const r = node.getBoundingClientRect();
      showTip(html(), r.right, r.top);
    });
    node.addEventListener("blur", hideTip);
  }

  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ---------- sections ---------- */

  function renderStats(r) {
    const s = r.summary;
    const high = s.levels.high;
    const tiles = [
      [pct(s.accuracy), "Accuracy on held-out images", `${fmt(s.test_images)} images not used for training or threshold fitting`],
      [pct(s.macro_class_accuracy), "Average accuracy per class", "Each of the 38 classes counts equally"],
      [pct(high.accuracy), "Accuracy at “High” confidence", `${pct(high.coverage, 0)} of images reach this level`],
      [pct(s.ece), "Calibration error (ECE)", "Gap between stated confidence and actual accuracy"],
      [s.model_size_mb != null ? `${s.model_size_mb} MB` : "—", "Deployed model size", "int8 TensorFlow Lite"],
    ];

    const grid = $("statGrid");
    grid.innerHTML = "";
    tiles.forEach(([value, label, sub]) => {
      const tile = el("div", { class: "stat-tile glass" });
      tile.appendChild(el("div", { class: "stat-tile-value" }, value));
      tile.appendChild(el("div", { class: "stat-tile-label" }, label));
      tile.appendChild(el("div", { class: "stat-tile-sub" }, sub));
      grid.appendChild(tile);
    });
  }

  function renderLevels(r) {
    const s = r.summary;
    const rules = {
      high: `≥ ${pct(s.thresholds.high)}`,
      uncertain: `${pct(s.thresholds.low)} – ${pct(s.thresholds.high)}`,
      low: `< ${pct(s.thresholds.low)} (no diagnosis shown)`,
    };
    const labels = { high: "✓ High", uncertain: "! Uncertain", low: "✕ Low" };
    const body = $("levelsBody");
    body.innerHTML = "";

    ["high", "uncertain", "low"].forEach((level) => {
      const row = s.levels[level];
      const tr = el("tr");
      const name = el("td");
      name.appendChild(el("span", { class: `level-pill level-${level}` }, labels[level]));
      tr.appendChild(name);
      tr.appendChild(el("td", {}, rules[level]));
      tr.appendChild(el("td", { class: "num" }, pct(row.coverage)));
      tr.appendChild(el("td", { class: "num" }, row.accuracy == null ? "—" : pct(row.accuracy)));
      tr.appendChild(
        el("td", { class: "num" }, row.accuracy_95ci ? `${pct(row.accuracy_95ci[0])} – ${pct(row.accuracy_95ci[1])}` : "—")
      );
      body.appendChild(tr);
    });
  }

  function renderClassBars(r) {
    const rows = [...r.per_class].sort((a, b) => a.accuracy - b.accuracy);
    const bars = $("classBars");
    const tbody = $("classTableBody");
    bars.innerHTML = "";
    tbody.innerHTML = "";

    rows.forEach((c) => {
      const row = el("div", { class: "class-bar", role: "listitem", tabindex: "0" });
      row.appendChild(el("span", { class: "class-bar-label" }, c.display_name));
      const track = el("span", { class: "class-bar-track" });
      const fill = el("span", { class: "class-bar-fill" });
      fill.style.width = `${(c.accuracy * 100).toFixed(1)}%`;
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("span", { class: "class-bar-value" }, pct(c.accuracy)));
      row.setAttribute("aria-label", `${c.display_name}: ${pct(c.accuracy)} accuracy`);
      bindTip(row, () =>
        `<strong>${escapeHtml(c.display_name)}</strong><br>Accuracy ${pct(c.accuracy)}<br>` +
        `Test images ${fmt(c.test_images)} · Training images ${fmt(c.train_images)}`
      );
      bars.appendChild(row);

      const tr = el("tr");
      tr.appendChild(el("td", {}, c.display_name));
      tr.appendChild(el("td", { class: "num" }, pct(c.accuracy)));
      tr.appendChild(el("td", { class: "num" }, fmt(c.test_images)));
      tr.appendChild(el("td", { class: "num" }, fmt(c.train_images)));
      tbody.appendChild(tr);
    });
  }

  function rampColor(t) {
    const index = Math.min(SEQ_RAMP.length - 1, Math.floor(t * (SEQ_RAMP.length - 1) + 0.5));
    return SEQ_RAMP[index];
  }

  function renderConfusion(r) {
    const m = r.confusion_matrix;
    const names = r.class_display_names;
    const n = m.length;
    const cell = 12;
    const gap = 1;
    const axis = 30;
    const size = axis + n * cell;

    // Row-normalised error rates (diagonal excluded).
    const rates = m.map((row, i) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      return row.map((v, j) => (i === j ? null : v / total));
    });
    const max = Math.max(...rates.flat().filter((v) => v != null), 0.0001);

    const svg = $("confusionSvg");
    const NS = "http://www.w3.org/2000/svg";
    svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
    svg.innerHTML = "";

    const label = (x, y, text, anchor = "middle") => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", x);
      t.setAttribute("y", y);
      t.setAttribute("text-anchor", anchor);
      t.setAttribute("class", "axis-label");
      t.textContent = text;
      svg.appendChild(t);
    };

    label(axis + (n * cell) / 2, 18, "Predicted class →");
    const yTitle = document.createElementNS(NS, "text");
    yTitle.setAttribute("transform", `translate(18 ${axis + (n * cell) / 2}) rotate(-90)`);
    yTitle.setAttribute("text-anchor", "middle");
    yTitle.setAttribute("class", "axis-label");
    yTitle.textContent = "True class →";
    svg.appendChild(yTitle);

    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const rect = document.createElementNS(NS, "rect");
        rect.setAttribute("x", axis + j * cell + gap / 2);
        rect.setAttribute("y", axis + i * cell + gap / 2);
        rect.setAttribute("width", cell - gap);
        rect.setAttribute("height", cell - gap);
        rect.setAttribute("rx", 1.5);
        const v = rates[i][j];
        rect.setAttribute("fill", v == null ? DIAGONAL_CELL : v === 0 ? ZERO_CELL : rampColor(v / max));
        rect.setAttribute("class", "cm-cell");
        bindTip(rect, () =>
          v == null
            ? `<strong>${escapeHtml(names[i])}</strong><br>Correct: ${fmt(m[i][j])} images`
            : `True: <strong>${escapeHtml(names[i])}</strong><br>Predicted: <strong>${escapeHtml(names[j])}</strong><br>` +
              `${fmt(m[i][j])} images (${pct(v)} of this class)`
        );
        svg.appendChild(rect);
      }
    }

    $("scaleMax").textContent = pct(max);

    const body = $("confusionBody");
    body.innerHTML = "";
    r.top_confusions.slice(0, 10).forEach((c) => {
      const tr = el("tr");
      tr.appendChild(el("td", {}, c.true_display));
      tr.appendChild(el("td", {}, c.predicted_display));
      tr.appendChild(el("td", { class: "num" }, fmt(c.count)));
      body.appendChild(tr);
    });
  }

  const FIGURES = {
    reliability_diagram: "Reliability diagram — stated confidence vs. actual accuracy (closer to the diagonal is better).",
    accuracy_vs_coverage: "Accuracy vs. coverage — accepting fewer, more confident images raises accuracy. Dashed lines mark the High and Low thresholds.",
    training_accuracy: "Training vs. validation accuracy per epoch.",
    training_loss: "Training vs. validation loss per epoch.",
  };

  function renderFigures(r) {
    const grid = $("figureGrid");
    grid.innerHTML = "";
    Object.keys(FIGURES)
      .filter((name) => r.figures.includes(name))
      .forEach((name) => {
        const fig = el("figure", { class: "perf-figure" });
        const img = el("img", { src: `/model/figures/${name}`, alt: FIGURES[name], loading: "lazy" });
        fig.appendChild(img);
        fig.appendChild(el("figcaption", {}, FIGURES[name]));
        grid.appendChild(fig);
      });
  }

  function renderComparison(r) {
    const o = r.comparison_original;
    if (!o) return;
    const d = r.summary;
    $("comparisonCard").classList.remove("hidden");
    const rows = [
      ["Model size", o.model_size_mb != null ? `${o.model_size_mb} MB` : "—", d.model_size_mb != null ? `${d.model_size_mb} MB` : "—"],
      ["Accuracy (held-out)", pct(o.accuracy, 2), pct(d.accuracy, 2)],
      ["Average per-class accuracy", pct(o.macro_class_accuracy, 2), pct(d.macro_class_accuracy, 2)],
      ["Accuracy at High level", pct(o.levels.high.accuracy, 2), pct(d.levels.high.accuracy, 2)],
      ["Share of images at High level", pct(o.levels.high.coverage), pct(d.levels.high.coverage)],
      ["Calibration error (ECE)", pct(o.ece, 2), pct(d.ece, 2)],
    ];
    const body = $("comparisonBody");
    body.innerHTML = "";
    rows.forEach(([name, a, b]) => {
      const tr = el("tr");
      tr.appendChild(el("td", {}, name));
      tr.appendChild(el("td", { class: "num" }, a));
      tr.appendChild(el("td", { class: "num" }, b));
      body.appendChild(tr);
    });
  }

  function renderLimits(r) {
    const weakest = [...r.per_class].sort((a, b) => a.accuracy - b.accuracy).slice(0, 3);
    const q = r.quality_flags;
    const items = [
      r.provisional_reason,
      `The held-out images come from the PlantVillage validation split, which was also used for early stopping during training, so these numbers are somewhat optimistic.`,
      `Weakest classes: ${weakest.map((c) => `${c.display_name} (${pct(c.accuracy)})`).join(", ")}.`,
      "Explanation heat maps on sample images show the model partly relies on the background and leaf edges, not only on the symptoms — a known weakness of models trained on lab photos with plain backgrounds.",
    ];
    if (q) {
      items.push(
        `Image-quality warnings fire on ${pct(q.flagged_fraction)} of held-out lab images; on these lab images flagged photos were not less accurate (${pct(q.accuracy_flagged)} vs ${pct(q.accuracy_unflagged)}). The warnings are kept as a safeguard for poor real-world photos, where their effect has not been measured.`
      );
    }
    const list = $("limitsList");
    list.innerHTML = "";
    items.forEach((text) => list.appendChild(el("li", {}, text)));
  }

  async function load() {
    try {
      const response = await fetch("/model/report");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not load the report.");

      $("perfIntro").textContent =
        `Evaluation of the deployed model on ${fmt(data.summary.test_images)} held-out PlantVillage images (results from ${data.created}).`;
      $("provisionalNote").textContent =
        "These numbers come from lab photos (single leaves, plain backgrounds). Accuracy on real-world field photos has not been measured and is expected to be lower.";

      renderStats(data);
      renderLevels(data);
      renderClassBars(data);
      renderConfusion(data);
      renderFigures(data);
      renderComparison(data);
      renderLimits(data);

      $("perfContent").classList.remove("hidden");
    } catch (err) {
      $("perfIntro").textContent = "";
      const box = $("perfError");
      box.querySelector("p").textContent = err.message || "Could not load the report.";
      box.classList.remove("hidden");
    }
  }

  load();
})();
