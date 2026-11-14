const LABEL = {
  ada_flag: "ADA risk test", ada_flag_asian23: "ADA test, BMI 23+ for Asian adults", ada_flag_selfbmi: "ADA test, self-reported BMI",
  uspstf_flag: "USPSTF 2021 rule", uspstf_flag_asian23: "USPSTF, BMI 23+ for Asian adults", ada2026_flag: "ADA 2026 rule",
  ada_points: "ADA risk test (points)", ada_refit: "Refitted ADA items", l1: "L1 logistic", ebm: "EBM",
  ebm_pairs: "EBM with pairs", lgb: "LightGBM classifier", lgb_value: "LightGBM value model",
};
const COLOR = {
  ada_flag: "#6b7a90", ada_points: "#6b7a90", uspstf_flag: "#9c6644", uspstf_flag_asian23: "#c99a74",
  ada2026_flag: "#8a5a9e", ada_flag_asian23: "#a7b1c2", ada_refit: "#a7b1c2", l1: "#c98a0b", ebm: "#2f6fde",
  ebm_pairs: "#8fb0ec", lgb: "#0e7470", lgb_value: "#5fb3ad",
};
const OPS = { ada: "ADA risk test", uspstf: "USPSTF 2021", ada2026: "ADA 2026" };
const MODELS = ["ada_refit", "l1", "ebm", "ebm_pairs", "lgb", "lgb_value"];
const FONT = { family: "Public Sans, system-ui, sans-serif", size: 13, color: "#17243b" };
const CONFIG = { displayModeBar: false, responsive: true };

const pct = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x) ? "–" : `${(100 * x).toFixed(d)}%`);
const num = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const signed = (x, d = 1) => {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const v = Number(Number(x).toFixed(d));
  return v === 0 ? (0).toFixed(d) : `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
};
const pts = (x, d = 1) => (x === null || x === undefined ? "–" : signed(100 * x, d));
const NARROW = window.matchMedia("(max-width: 760px)");
const wrap = (s, n) => s.split(" ").reduce((lines, word) => {
  const i = lines.length - 1;
  if (i >= 0 && `${lines[i]} ${word}`.length <= n) lines[i] = `${lines[i]} ${word}`;
  else lines.push(word);
  return lines;
}, []).join("<br>");
const STEP = {
  ada_refit: "re-weighting the ADA items", l1: "giving a linear model all the inputs instead of the seven ADA items",
  ebm: "learned curves for each input", ebm_pairs: "pairwise interactions", lgb: "boosted trees with any interaction",
};
const cyc = (c) => c.replace(/^(\d{4})-20(\d\d)$/, "$1-$2");
const $ = (id) => document.getElementById(id);

function layout(extra) {
  return Object.assign({
    font: FONT, paper_bgcolor: "#fff", plot_bgcolor: "#fff", margin: { l: 60, r: 20, t: 30, b: 50 },
    xaxis: { gridcolor: "#e6e9ee", zeroline: false }, yaxis: { gridcolor: "#e6e9ee", zeroline: false },
    showlegend: false, hoverlabel: { font: FONT },
  }, extra);
}

function table(el, head, rows, cls = () => "") {
  const h = `<thead><tr>${head.map((x) => `<th scope="col">${x}</th>`).join("")}</tr></thead>`;
  const isNum = (c) => /^[-–−+]?[\d.,]+%?(\s|$|<)/.test(String(c)) || String(c) === "–";
  const b = rows.map((r, i) => `<tr class="${cls(i)}">${r.map((c, j) => `<td class="${j > 0 && isNum(c) ? "n" : ""}">${c}</td>`).join("")}</tr>`).join("");
  el.innerHTML = `<table>${h}<tbody>${b}</tbody></table>`;
}

async function getJSON(name) {
  const res = await fetch(`data/${name}`);
  return res.ok ? res.json() : null;
}

function renderFlow(flow) {
  const cols = flow.cycles.map((c) => (c === "Total" ? c : cyc(c)));
  table($("flow-table"), ["Step", ...cols], flow.steps.map((s, i) => [s.trim(), ...flow.counts[i].map((n) => n.toLocaleString("en-US"))]),
    (i) => (flow.steps[i].startsWith(" ") ? "rule" : ""));
}

function renderValidation(rows) {
  const order = ["ada_points", ...MODELS];
  const get = (t, m) => rows.find((r) => r.target === t && r.model === m) || {};
  table($("validation-table"), ["Method", "AUC 5.7%+", "Sensitivity at ADA test's share tested", "AUC 6.5%+", "Observed / expected 5.7%+"],
    order.map((m) => [LABEL[m], num(get("y57", m).auc, 3), pct(get("y57", m).sens_at_ada_rate, 1), num(get("y65", m).auc, 3),
      m === "ada_points" ? "–" : num(get("y57", m).obs_exp, 2)]),
    (i) => (i === 0 ? "rule" : ""));
}

function renderLadder(rows) {
  const short = ["Honest: later survey, undiagnosed adults", "Random split instead of a later survey", "+ diagnosed adults counted as cases",
    "+ two questions that reveal past testing", "+ one blood value (HDL)"];
  const narrow = NARROW.matches;
  const y = rows.map((r, i) => (narrow ? wrap(short[i] || r.step, 22) : short[i] || r.step)).reverse();
  const x = rows.map((r) => r.auc).reverse();
  Plotly.newPlot("ladder-chart", [{
    type: "bar", orientation: "h", x, y, marker: { color: x.map((_, i) => (i === x.length - 1 ? "#0e7470" : "#a7b1c2")) },
    text: x.map((v) => v.toFixed(3)), textposition: "outside", cliponaxis: false,
    hovertemplate: "%{y}<br>AUC %{x:.3f}<extra></extra>",
  }], layout({
    margin: { l: narrow ? 160 : 290, r: 50, t: 10, b: 40 }, height: narrow ? 340 : undefined,
    xaxis: { range: [0.7, 0.9], title: narrow ? "AUC, HbA1c 5.7%+" : "AUC for HbA1c 5.7%+ on held-out data", nticks: narrow ? 3 : undefined, gridcolor: "#e6e9ee", zeroline: false },
  }), CONFIG);
}

function row(R, t, op, m) {
  return R.main_table.find((r) => r.target === t && r.operating_point === op && r.method === m);
}

function renderHero(R, Y) {
  const us = row(R, "y57", "uspstf", "uspstf_flag");
  const rule = row(R, "y57", "ada", "ada_flag");
  const model = row(R, "y57", "ada", "lgb");
  const cases = Math.round(100 * Y.y57.prevalence);
  const panels = [["USPSTF 2021 rule", us], ["ADA risk test", rule], ["LightGBM model", model]];
  $("waffles").innerHTML = panels.map(([name, r]) => {
    const tested = Math.round(100 * r.flag_rate);
    const found = Math.round(cases * r.sens);
    const dots = [];
    for (let i = 0; i < 100; i += 1) {
      if (i < found) dots.push("found");
      else if (i < cases) dots.push("missed");
      else if (i < cases + tested - found) dots.push("tested");
      else dots.push("");
    }
    return `<div class="waffle"><h3>${name}</h3><p class="sub">Tests ${tested} of every 100 adults and finds ${found} of the ${cases} with HbA1c 5.7% or more.</p>
      <div class="grid" role="img" aria-label="${name}: ${found} of ${cases} cases found, ${tested} tested per 100 adults">${dots.map((d) => `<span class="dot ${d}"></span>`).join("")}</div></div>`;
  }).join("");
  $("waffle-legend").hidden = false;
  const h = R.hypotheses.H1_lgb;
  const e = R.hypotheses.H1_ebm;
  const nb = R.hypotheses.H1_lgb_without_borderline;
  const per100 = (r) => Math.round(100 * r.flag_rate);
  $("hero-lead").textContent = `About ${cases} in 100 US adults never told they have diabetes had an HbA1c of 5.7% or more in 2017-2020. `
    + `Sending ${per100(rule)} in 100 adults for a blood test, the ADA's seven-question risk test found ${pct(rule.sens)} of those ${cases}. `
    + `A model trained on the 2007-2014 surveys sent ${per100(model)} in 100 and found ${pct(model.sens)}, ${h.supported ? "a clear gain" : "a gain too small to tell apart from chance"}. `
    + `The USPSTF 2021 rule sent ${per100(us)} in 100 and found ${pct(us.sens)}.`;
  $("h1-callout").innerHTML = `<strong>${h.supported ? "H1 supported: the model finds more cases for the same number of tests." : "H1 not supported: the model's gain is too small to tell apart from zero."}</strong>`
    + `With its threshold set on 2015-2016 to test as many adults as the ADA risk test, the LightGBM classifier found ${pct(h.sens_model, 1)} of adults with HbA1c of 5.7% or more in 2017-2020, against ${pct(h.sens_ada, 1)} for the ADA test: `
    + `${pts(h.diff)} points (95% interval ${pts(h.lo)} to ${pts(h.hi)}). This was the one confirmatory test fixed in the pre-registration. `
    + `The EBM (${pts(e.diff)} points, ${pts(e.lo)} to ${pts(e.hi)}) and the same test without adults who answered "borderline" (${pts(nb.diff)}, ${pts(nb.lo)} to ${pts(nb.hi)}) agree.`;
}

function renderOps(R) {
  const methods = ["ada_refit", "l1", "ebm", "ebm_pairs", "lgb", "lgb_value"];
  const ruleOf = { ada: "ada_flag", uspstf: "uspstf_flag", ada2026: "ada2026_flag" };
  const traces = [];
  const ops = Object.keys(OPS);
  const narrow = NARROW.matches;
  ops.forEach((op, k) => {
    const names = [ruleOf[op], ...methods];
    const rows = names.map((m) => row(R, "y57", op, m));
    traces.push({
      type: "scatter", mode: "markers", xaxis: k === 0 ? "x" : `x${k + 1}`, yaxis: narrow && k > 0 ? `y${k + 1}` : "y",
      x: rows.map((r) => r.sens), y: names.map((m) => (m.endsWith("_flag") ? "The rule itself" : LABEL[m])),
      error_x: { type: "data", symmetric: false, array: rows.map((r) => r.sens_hi - r.sens), arrayminus: rows.map((r) => r.sens - r.sens_lo), color: "#7a8494", thickness: 1.5, width: 0 },
      marker: { size: 10, color: names.map((m) => COLOR[m]) },
      hovertemplate: rows.map((r, i) => `${LABEL[names[i]]}<br>sensitivity ${pct(r.sens, 1)} (${pct(r.sens_lo, 1)} to ${pct(r.sens_hi, 1)})<br>share tested ${pct(r.flag_rate, 1)}<extra></extra>`),
    });
  });
  const lay = layout({ margin: { l: narrow ? 150 : 170, r: 20, t: narrow ? 30 : 40, b: 50 }, yaxis: { autorange: "reversed", gridcolor: "#e6e9ee" }, annotations: [] });
  if (narrow) lay.height = 860;
  const n = ops.length;
  ops.forEach((op, k) => {
    const rs = [ruleOf[op], ...methods].map((m) => row(R, "y57", op, m));
    const span = Math.max(...rs.map((r) => r.sens_hi)) - Math.min(...rs.map((r) => r.sens_lo));
    const xa = { tickformat: span < 0.05 ? ".1%" : ".0%", gridcolor: "#e6e9ee", zeroline: false, title: !narrow || k === n - 1 ? "sensitivity" : "" };
    const text = `${OPS[op]}: ${pct(rs[0].flag_rate)} tested`;
    if (narrow) {
      const dom = [1 - (k + 1) / n + 0.075, 1 - k / n - 0.035];
      lay[k === 0 ? "xaxis" : `xaxis${k + 1}`] = { ...xa, domain: [0, 1], anchor: k === 0 ? "y" : `y${k + 1}`, nticks: 5 };
      lay[k === 0 ? "yaxis" : `yaxis${k + 1}`] = { domain: dom, anchor: k === 0 ? "x" : `x${k + 1}`, autorange: "reversed", gridcolor: "#e6e9ee", zeroline: false };
      lay.annotations.push({ text, xref: "paper", yref: "paper", x: 0.5, y: dom[1] + 0.008, yanchor: "bottom", showarrow: false, font: { size: 13 } });
    } else {
      const w = 1 / n;
      lay[k === 0 ? "xaxis" : `xaxis${k + 1}`] = { ...xa, domain: [k * w + 0.02, (k + 1) * w - 0.02] };
      lay.annotations.push({ text, xref: "paper", yref: "paper", x: k * w + w / 2, y: 1.08, showarrow: false, font: { size: 13 } });
    }
  });
  Plotly.newPlot("ops-chart", traces, lay, CONFIG);
}

function renderYield(Y) {
  const draw = () => {
    const t = $("target").value;
    const rate = Number($("rate").value) / 100;
    $("rate-out").textContent = `${Math.round(rate * 100)}%`;
    const Yt = Y[t];
    const show = ["ada_points", "l1", "ebm", "lgb", "lgb_value"];
    const traces = show.map((m) => ({
      type: "scatter", mode: "lines", name: LABEL[m], x: Yt.curves[m].map((p) => p[1]), y: Yt.curves[m].map((p) => p[2]),
      line: { color: COLOR[m], width: m === "lgb" ? 3 : 2, shape: m === "ada_points" ? "hv" : "linear" },
      hovertemplate: `${LABEL[m]}<br>tested %{x:.0%}, sensitivity %{y:.1%}<extra></extra>`,
    }));
    const rules = ["ada_flag", "uspstf_flag", "ada2026_flag"];
    const narrow = NARROW.matches;
    const offset = narrow ? { ada_flag: [-60, -30], uspstf_flag: [50, 60], ada2026_flag: [-40, 80] } : { ada_flag: [95, 12], uspstf_flag: [95, 50], ada2026_flag: [-70, 55] };
    rules.forEach((m) => {
      const [fr, se] = Yt.rules[m];
      traces.push({ type: "scatter", mode: "markers", name: LABEL[m], showlegend: false, x: [fr], y: [se], cliponaxis: false,
        marker: { size: 11, color: COLOR[m], symbol: "diamond", line: { color: "#fff", width: 1 } },
        hovertemplate: `${LABEL[m]}<br>tested %{x:.0%}, sensitivity %{y:.1%}<extra></extra>` });
    });
    const annotations = rules.map((m) => ({
      x: Yt.rules[m][0], y: Yt.rules[m][1], text: LABEL[m], ax: offset[m][0], ay: offset[m][1], showarrow: true, arrowhead: 0,
      arrowwidth: 1, arrowcolor: "#4a5568", font: { size: 12, color: "#17243b" }, bgcolor: "rgba(255,255,255,0.9)",
    }));
    Plotly.react("yield-chart", traces, layout({
      showlegend: true, annotations,
      legend: narrow ? { orientation: "h", x: 0, y: -0.25, yanchor: "top" } : { orientation: "h", x: 0, y: 1.02, yanchor: "bottom" },
      margin: narrow ? { l: 55, r: 15, t: 20, b: 200 } : { l: 60, r: 20, t: 50, b: 60 }, height: narrow ? 560 : undefined,
      xaxis: { title: "share of adults sent for an HbA1c test", tickformat: ".0%", range: [0, 0.92], gridcolor: "#e6e9ee", zeroline: false },
      yaxis: { title: "share of cases found", tickformat: ".0%", range: [0, 1.02], gridcolor: "#e6e9ee", zeroline: false },
      shapes: [{ type: "line", x0: rate, x1: rate, y0: 0, y1: 1, line: { color: "#17243b", width: 1, dash: "dot" } }],
    }), CONFIG);
    const idx = Math.round(rate * 100) - 1;
    table($("yield-table"), ["Method", "Share tested", "Cases found", "Tested adults who are cases", "Tests per case found"],
      show.map((m) => {
        const [, fr, se, pp] = Yt.curves[m][idx];
        return [LABEL[m], pct(fr), pct(se, 1), pct(pp, 1), num(1 / pp, 1)];
      }));
  };
  ["rate", "target"].forEach((id) => $(id).addEventListener("input", draw));
  draw();
}

function renderMain(R) {
  const head = ["Method", "Share tested", "Sensitivity (95% CI)", "PPV", "Tests per case", "Gain vs rule, points (95% CI)", "Sensitivity, threshold re-set on 2017-20"];
  const p1 = (x) => pct(x, 1);
  const ci = (a, b, f) => ` <span class="ci">(${f(a)} to ${f(b)})</span>`;
  $("main-table").innerHTML = ["y57", "y65"].map((t) => {
    const body = Object.keys(OPS).map((op) => {
      const rows = R.main_table.filter((r) => r.target === t && r.operating_point === op).map((r) => {
        const isRule = r.method.endsWith("_flag");
        const cells = [pct(r.flag_rate), `${p1(r.sens)}${ci(r.sens_lo, r.sens_hi, p1)}`, p1(r.ppv), num(r.tests_per_case, 1),
          isRule ? "–" : `${pts(r.sens_diff)}${ci(r.sens_diff_lo, r.sens_diff_hi, pts)}`, isRule ? "–" : p1(r.sens_matched_on_test)];
        return `<tr><th scope="row">${LABEL[r.method]}</th>${cells.map((c) => `<td class="n">${c}</td>`).join("")}</tr>`;
      }).join("");
      return `<tr class="group"><th colspan="${head.length}" scope="colgroup">Number of tests set by the ${LABEL[`${op}_flag`]}</th></tr>${rows}`;
    }).join("");
    return `<table><caption>HbA1c ${t === "y57" ? "5.7" : "6.5"}% or more</caption><thead><tr>${head.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;
  }).join("");
}

function renderDisc(R) {
  const rows = R.discrimination.map((r) => [r.target === "y57" ? "5.7%+" : "6.5%+", LABEL[r.method],
    `${num(r.auc, 3)} <span class="ci">(${num(r.auc_lo, 3)} to ${num(r.auc_hi, 3)})</span>`, num(r.pr_auc, 3),
    r.brier === undefined || r.brier === null ? "–" : num(r.brier, 4), r.obs_exp ? num(r.obs_exp, 2) : "–", r.cal_slope ? num(r.cal_slope, 2) : "–"]);
  table($("disc-table"), ["Cut-off", "Method", "AUC (95% CI)", "PR-AUC", "Brier", "Observed / expected", "Calibration slope"], rows);
}

function renderH(R) {
  const H = R.hypotheses;
  table($("h3-table"), ["Step", "What it adds", "Gain in sensitivity, points (95% CI)"], H.H3.map((s) => [`${LABEL[s.from]} to ${LABEL[s.to]}`, STEP[s.to],
    `${pts(s.gain)} <span class="ci">(${pts(s.lo)} to ${pts(s.hi)})</span>`]));
  const clear = H.H3.filter((s) => s.lo > 0 || s.hi < 0);
  const rest = H.H3.filter((s) => !(s.lo > 0 || s.hi < 0));
  const gains = rest.map((s) => s.gain);
  $("h3-text").textContent = (clear.length
    ? `Only ${clear.map((s) => `${STEP[s.to]} (${pts(s.gain)} points, ${pts(s.lo)} to ${pts(s.hi)})`).join(" and ")} changed sensitivity by more than the noise. `
    : "No single step changed sensitivity by more than the noise. ")
    + (rest.length ? `The other steps moved it by between ${pts(Math.min(...gains))} and ${pts(Math.max(...gains))} points, with intervals that include zero.` : "");
  const h2 = H.H2;
  const n65 = row(R, "y65", "ada", "ada_flag").cases;
  $("h2-text").textContent = `For HbA1c of 6.5% or more, predicting the HbA1c value (model B) instead of the yes/no label (model A) changed PR-AUC by ${signed(h2.pr_auc_diff, 3)} `
    + `(95% interval ${signed(h2.pr_auc_lo, 3)} to ${signed(h2.pr_auc_hi, 3)}) and sensitivity at the ADA test's share tested by ${pts(h2.sens_diff)} points `
    + `(${pts(h2.sens_lo)} to ${pts(h2.sens_hi)}). There is no detectable difference, and with ${n65} cases in the test survey the comparison is imprecise.`;
}

function renderCycles(R) {
  const H = R.hba1c_by_cycle;
  Plotly.newPlot("hba1c-cycle-chart", [{
    type: "scatter", mode: "lines+markers", x: H.map((r) => cyc(r.cycle)), y: H.map((r) => r.pct_57 / 100),
    line: { color: "#c98a0b", width: 2 }, marker: { size: 8 }, hovertemplate: "%{x}: %{y:.1%}<extra></extra>",
  }], layout({ xaxis: { type: "category", gridcolor: "#e6e9ee", automargin: true }, yaxis: { tickformat: ".0%", title: "HbA1c 5.7%+", gridcolor: "#e6e9ee" }, margin: { l: 60, r: 20, t: 20, b: 40 } }), CONFIG);
  const draw = () => {
    const m = $("cycle-model").value;
    const rows = R.calibration_by_cycle.filter((r) => r.model === m && r.target === "y57");
    const raw = rows.filter((r) => r.kind.startsWith("uncalibrated")).sort((a, b) => a.cycle.localeCompare(b.cycle));
    const cal = rows.filter((r) => r.kind.startsWith("calibrated"));
    Plotly.react("oe-cycle-chart", [
      { type: "scatter", mode: "lines+markers", name: "before recalibration", x: raw.map((r) => cyc(r.cycle)), y: raw.map((r) => r.obs_exp), line: { color: "#7a8494" }, marker: { size: 8 } },
      { type: "scatter", mode: "markers", name: "recalibrated on 2015-16", x: cal.map((r) => cyc(r.cycle)), y: cal.map((r) => r.obs_exp), marker: { size: 12, color: "#0e7470", symbol: "diamond" } },
    ], layout({
      showlegend: true, legend: { orientation: "h", y: -0.25 }, margin: { l: 60, r: 10, t: 20, b: 70 },
      xaxis: { type: "category", categoryorder: "array", categoryarray: ["2011-12", "2013-14", "2015-16", "2017-20"], gridcolor: "#e6e9ee" },
      yaxis: { title: "observed / expected", gridcolor: "#e6e9ee" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: "#17243b", width: 1, dash: "dot" } }],
    }), CONFIG);
  };
  $("cycle-model").addEventListener("input", draw);
  draw();
  const cb = R.calibration_by_cycle;
  const raw = (t) => cb.filter((r) => r.target === t && r.kind.startsWith("uncalibrated"));
  const lo = (xs) => num(Math.min(...xs), 2);
  const hi = (xs) => num(Math.max(...xs), 2);
  const oe57 = raw("y57").map((r) => r.obs_exp);
  const oe65 = raw("y65").map((r) => r.obs_exp);
  const slopes = raw("y57").concat(raw("y65")).map((r) => r.cal_slope);
  const cal = (t) => R.discrimination.find((r) => r.target === t && r.method === "lgb");
  const p65 = (c) => H.find((r) => r.cycle === c).pct_65;
  const tp = R.train_prevalence.y57;
  $("h4-text").textContent = `Before recalibration, every model expected more cases than were observed, in every later survey: observed/expected ${lo(oe57)} to ${hi(oe57)} for HbA1c 5.7%+ `
    + `and ${lo(oe65)} to ${hi(oe65)} for 6.5%+. The likely reason is that the models were trained without survey weights: ${pct(tp.unweighted)} of the 2007-2014 sample has HbA1c 5.7%+, `
    + `but ${pct(tp.weighted)} once weighted to US adults. Recalibrating on 2015-2016 brought 2017-2020 to ${num(cal("y57").obs_exp, 2)} for LightGBM (95% interval ${num(cal("y57").obs_exp_lo, 2)} to ${num(cal("y57").obs_exp_hi, 2)}). `
    + `For 6.5%+ the share rose from ${num(p65("2015-2016"), 1)}% to ${num(p65("2017-2020"), 1)}% between those two surveys, and the recalibrated LightGBM expected fewer diabetes-range cases than were found `
    + `(observed/expected ${num(cal("y65").obs_exp, 2)}, ${num(cal("y65").obs_exp_lo, 2)} to ${num(cal("y65").obs_exp_hi, 2)}). Calibration slopes ranged from ${lo(slopes)} to ${hi(slopes)}, where 1 is ideal.`;
}

function renderSubgroups(R) {
  const draw = () => {
    const dim = $("dim").value;
    const methods = dim === "race" ? ["ada_flag", "ada_flag_asian23", "uspstf_flag", "uspstf_flag_asian23", "ebm", "lgb"] : ["ada_flag", "uspstf_flag", "ebm", "lgb"];
    const rows = R.subgroups.filter((r) => r.dimension === dim);
    const groups = [...new Set(rows.map((r) => r.group))].sort();
    const traces = methods.map((m, k) => {
      const rs = groups.map((g) => rows.find((r) => r.group === g && r.method === m));
      const off = (k - (methods.length - 1) / 2) * 0.13;
      return {
        type: "scatter", mode: "markers", name: LABEL[m], x: rs.map((r) => r.sens), y: groups.map((_, i) => i + off),
        error_x: { type: "data", symmetric: false, array: rs.map((r) => (Number.isFinite(r.sens_hi) ? r.sens_hi - r.sens : 0)), arrayminus: rs.map((r) => (Number.isFinite(r.sens_lo) ? r.sens - r.sens_lo : 0)), color: COLOR[m], thickness: 1.2, width: 0 },
        marker: { size: 9, color: COLOR[m] },
        hovertemplate: rs.map((r) => `${LABEL[m]}, ${r.group}<br>sensitivity ${pct(r.sens, 1)} (${pct(r.sens_lo, 1)} to ${pct(r.sens_hi, 1)})<br>${r.cases} cases among ${r.n} adults<extra></extra>`),
      };
    });
    const narrow = NARROW.matches;
    Plotly.react("subgroup-chart", traces, layout({
      showlegend: true, legend: { orientation: "h", y: -0.15 }, margin: { l: narrow ? 125 : 230, r: 20, t: 10, b: 90 },
      xaxis: { title: "share of cases found", tickformat: ".0%", range: [0, 1.02], gridcolor: "#e6e9ee", zeroline: false },
      yaxis: { tickvals: groups.map((_, i) => i), ticktext: groups.map((g) => { const r = rows.find((x) => x.group === g); const t = `${g} (${r.cases} cases)`; return narrow ? wrap(t, 18) : t; }), autorange: "reversed", gridcolor: "#fff", zeroline: false },
      height: 160 + groups.length * 70,
    }), CONFIG);
  };
  $("dim").addEventListener("input", draw);
  draw();
  const asian = (m) => R.subgroups.find((r) => r.dimension === "race" && r.group === "Non-Hispanic Asian" && r.method === m);
  if (asian("lgb")) {
    const white = R.subgroups.find((r) => r.dimension === "race" && r.group === "Non-Hispanic White" && r.method === "ada_flag");
    $("h5-text").textContent = `Asian adults are the clearest gap (${asian("lgb").cases} cases in the sample). The ADA test found ${pct(asian("ada_flag").sens)} of their cases with the BMI 25 cut-off `
      + `and ${pct(asian("ada_flag_asian23").sens)} with BMI 23; the LightGBM classifier found ${pct(asian("lgb").sens)} and the EBM ${pct(asian("ebm").sens)}, without any race input. `
      + `Among non-Hispanic White adults the ADA test found ${pct(white.sens)}. The USPSTF rule found ${pct(asian("uspstf_flag").sens)} with BMI 25 and ${pct(asian("uspstf_flag_asian23").sens)} with BMI 23, `
      + `but the BMI 23 version also tests ${pct(asian("uspstf_flag_asian23").flag_rate)} of Asian adults instead of ${pct(asian("uspstf_flag").flag_rate)}. `
      + "H5 holds in direction for every comparison, but only the USPSTF difference is clearly larger than the noise, and it comes with more tests.";
  }
}

function renderMissed(R) {
  const rows = R.missed_profile.filter((r) => r.target === "y57");
  const f = rows.find((r) => r.group === "found");
  const m = rows.find((r) => r.group === "missed");
  const items = [["Adults in the sample", (r) => r.n], ["Mean age", (r) => num(r.age, 1)], ["Mean BMI", (r) => num(r.bmi, 1)],
    ["Men", (r) => `${num(r.men_pct, 0)}%`], ["Mean HbA1c", (r) => `${num(r.hba1c, 2)}%`], ["Family history of diabetes", (r) => `${num(r.family_history_pct, 0)}%`],
    ["Told high blood pressure", (r) => `${num(r.told_high_bp_pct, 0)}%`], ["Non-Hispanic Asian", (r) => `${num(r.asian_pct, 0)}%`], ["Non-Hispanic Black", (r) => `${num(r.black_pct, 0)}%`]];
  table($("missed-table"), ["", "Sent for a test", "Missed"], items.map(([k, g]) => [k, g(f), g(m)]));
  const r65 = R.missed_profile.filter((r) => r.target === "y65");
  const f2 = r65.find((r) => r.group === "found");
  const m2 = r65.find((r) => r.group === "missed");
  $("missed-text").textContent = `Missed cases are younger (mean age ${num(m.age, 0)} against ${num(f.age, 0)}), leaner (BMI ${num(m.bmi, 1)} against ${num(f.bmi, 1)}) and far less often told they have high blood pressure `
    + `(${num(m.told_high_bp_pct, 0)}% against ${num(f.told_high_bp_pct, 0)}%), yet their HbA1c is about as high (${num(m.hba1c, 2)}% against ${num(f.hba1c, 2)}%). `
    + "Age accounts for about half of the LightGBM model's split gain and body size (waist-to-height ratio and BMI) for a fifth, so a young, lean adult looks low-risk whatever their HbA1c. "
    + `Among adults with HbA1c of 6.5% or more, the ${m2.n} in the sample that the model missed were also younger (${num(m2.age, 0)} against ${num(f2.age, 0)}) and leaner `
    + `(BMI ${num(m2.bmi, 1)} against ${num(f2.bmi, 1)}), and ${num(m2.asian_pct, 0)}% of them were Asian, against ${num(f2.asian_pct, 0)}% of those found.`;
}

function renderConclusion(R) {
  const h = R.hypotheses.H1_lgb;
  const d = R.discrimination.filter((r) => r.target === "y57");
  const auc = (m) => num(d.find((r) => r.method === m).auc, 3);
  const model = row(R, "y57", "ada", "lgb");
  const rule = row(R, "y57", "ada", "ada_flag");
  const us = row(R, "y57", "uspstf", "lgb");
  const usr = row(R, "y57", "uspstf", "uspstf_flag");
  const a26 = row(R, "y57", "ada2026", "ada2026_flag");
  const L = R.leakage_ladder;
  const paras = [
    `On a survey it had never seen, a LightGBM model built from the same kind of information as the ADA risk test, plus body measurements, found ${pct(h.sens_model, 1)} of adults with HbA1c of 5.7% or more, `
      + `against ${pct(h.sens_ada, 1)} for the ADA test at about the same number of tests. `
      + (h.supported
        ? `The difference, ${pts(h.diff)} points (95% interval ${pts(h.lo)} to ${pts(h.hi)}), supports the pre-registered hypothesis H1.`
        : `The difference, ${pts(h.diff)} points (95% interval ${pts(h.lo)} to ${pts(h.hi)}), is too small to tell apart from zero, so the pre-registered hypothesis H1 is not supported. The interval rules out a gain of more than about ${Math.ceil(100 * h.hi)} points.`),
    `The models' AUC was higher (${auc("lgb")} against ${auc("ada_points")} for the ADA points) and they wasted fewer tests: ${pct(model.ppv, 1)} of the adults LightGBM sent for a test had HbA1c of 5.7% or more, `
      + `against ${pct(rule.ppv, 1)} for the ADA test (${pts(model.ppv_diff)} points, ${pts(model.ppv_diff_lo)} to ${pts(model.ppv_diff_hi)}; a secondary result). `
      + "An EBM, a plain L1 logistic regression and gradient boosting did equally well. The only step on the methods ladder that clearly helped was giving a linear model all the inputs instead of the seven ADA items; learned curves and interactions added nothing measurable.",
    `The clearer result concerns today's screening rules. At the USPSTF 2021 rule's number of tests, LightGBM found ${pct(us.sens)} of cases against the rule's ${pct(usr.sens)} `
      + `(${pts(us.sens_diff)} points, ${pts(us.sens_diff_lo)} to ${pts(us.sens_diff_hi)}), and the ADA risk test alone found more cases than the USPSTF rule while testing no more adults. `
      + `The ADA 2026 rule finds ${pct(a26.sens)} of cases, but only by testing ${pct(a26.flag_rate)} of adults. Asian adults are the weak spot: the ADA test, the USPSTF rule and the models each found only about half of their cases.`,
    `Honest evaluation matters. On the same 2007-2016 data, loosening the setup (a random split, already-diagnosed adults counted as cases, two questions that reveal past testing) lifts AUC from ${num(L[0].auc, 3)} to ${num(L[3].auc, 3)}, `
      + "close to the 0.82-0.89 that published models report.",
    "For a clinic choosing whom to send for an HbA1c test, the seven-question ADA risk test already does about as well as a machine-learning model trained on 18,456 survey participants. The rule that leaves cases behind is the USPSTF age-and-BMI rule.",
  ];
  $("conclusion-text").innerHTML = paras.map((p) => `<p>${p}</p>`).join("");
}

const EBM = {};
function bin(cuts, v) {
  if (v === null || v === undefined || Number.isNaN(v)) return 0;
  let lo = 0;
  let hi = cuts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cuts[mid] <= v) lo = mid + 1; else hi = mid;
  }
  return lo + 1;
}
function ebmScore(m, x) {
  let s = m.intercept;
  const contrib = [];
  m.terms.forEach((t) => {
    let sc = t.scores;
    t.features.forEach((f, j) => { sc = sc[bin(t.cuts[j], x[m.features[f]])]; });
    contrib.push([t.name, sc]);
    s += sc;
  });
  const [a, b] = m.platt;
  return { p: 1 / (1 + Math.exp(-(a + b * s))), contrib };
}

function adaPoints(x, asian) {
  const parts = [];
  const age = x.age >= 60 ? 3 : x.age >= 50 ? 2 : x.age >= 40 ? 1 : 0;
  parts.push(["Age", age]);
  parts.push(["Man", x.male === 1 ? 1 : 0]);
  parts.push(["Diabetes during pregnancy", x.gdm === 1 ? 1 : 0]);
  parts.push(["Family history", x.famhx === 1 ? 1 : 0]);
  parts.push(["High blood pressure", x.htn_told === 1 ? 1 : 0]);
  parts.push(["Not physically active", x.active_rec === 0 ? 1 : 0]);
  const first = asian ? 23 : 25;
  parts.push(["Weight (BMI)", x.bmi >= 40 ? 3 : x.bmi >= 30 ? 2 : x.bmi >= first ? 1 : 0]);
  return parts;
}

const PRETTY = {
  age: "Age", male: "Sex", bmi: "BMI", waist: "Waist", whtr: "Waist-to-height", height: "Height", weight: "Weight", arm: "Arm",
  leg: "Leg length", htn_told: "High blood pressure", bp_med: "Blood-pressure medicine", famhx: "Family history", cvd: "Heart disease or stroke",
  gdm: "Diabetes in pregnancy", smoke_former: "Former smoker", smoke_current: "Current smoker", active_rec: "Leisure activity",
  mvpa_min_wk: "Activity minutes", sedentary_min: "Sitting time", educ: "Education", pir: "Income",
};

function readForm() {
  const f = new FormData($("calc-form"));
  const v = (k) => { const s = f.get(k); return s === null || s === "" ? NaN : Number(s); };
  const x = {};
  ["age", "male", "height", "weight", "waist", "htn_told", "bp_med", "famhx", "cvd", "gdm", "active_rec", "mvpa_min_wk", "sedentary_min", "educ"].forEach((k) => { x[k] = v(k); });
  if (x.male === 1) x.gdm = 0;
  x.bmi = x.weight / (x.height / 100) ** 2;
  const smoking = v("smoking");
  x.smoke_former = smoking === 1 ? 1 : 0;
  x.smoke_current = smoking === 2 ? 1 : 0;
  const filled = [];
  const lin = (k) => { const r = EBM.y57.fill[k]; return r.intercept + Object.entries(r.coef).reduce((s, [c, b]) => s + b * x[c], 0); };
  if (Number.isNaN(x.waist)) { x.waist = lin("waist"); filled.push("waist"); }
  x.whtr = x.waist / x.height;
  x.arm = lin("arm");
  x.leg = lin("leg");
  ["educ", "pir", "mvpa_min_wk", "sedentary_min"].forEach((k) => { if (Number.isNaN(x[k])) { x[k] = EBM.y57.median[k]; if (k !== "pir") filled.push(k); } });
  x.filled = filled;
  x.group = f.get("group");
  return x;
}

function runCalculator() {
  const x = readForm();
  if (!(x.age >= 20) || !(x.bmi > 10)) return;
  const asian = x.group === "asian";
  const r57 = ebmScore(EBM.y57, x);
  const r65 = ebmScore(EBM.y65, x);
  const parts = adaPoints(x, false);
  const points = parts.reduce((s, p) => s + p[1], 0);
  const uspstf = x.age >= 35 && x.age <= 70 && x.bmi >= (asian ? 23 : 25);
  const overweight = x.bmi >= (asian ? 23 : 25);
  const risk = x.famhx === 1 || (x.group && x.group !== "white") || x.cvd === 1 || x.htn_told === 1 || x.bp_med === 1 || x.active_rec === 0 || x.bmi >= 40;
  const ada2026 = x.age >= 35 || (overweight && risk) || x.gdm === 1;
  const top = r57.contrib.filter((c) => Math.abs(c[1]) > 0.005 && !["arm", "leg", "pir"].includes(c[0])).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 10).reverse();
  $("calc-out").innerHTML = `<div class="probs">
      <div class="prob"><b>${pct(r57.p)}</b><span>chance of HbA1c 5.7% or more (EBM)</span></div>
      <div class="prob"><b>${pct(r65.p, 1)}</b><span>chance of HbA1c 6.5% or more (EBM)</span></div></div>
    <ul class="rules-out">
      <li><b>ADA risk test: ${points} of 11 points</b>, ${points >= 5 ? "test recommended (5 or more)" : "below the cut-off of 5"} <span class="note">(${parts.filter((p) => p[1] > 0).map((p) => `${p[0]} ${p[1]}`).join(", ") || "no points"})</span></li>
      <li><b>USPSTF 2021:</b> ${uspstf ? "screening recommended" : "not in the screening group"}</li>
      <li><b>ADA 2026:</b> ${ada2026 ? "testing recommended" : "no testing recommended yet"}</li>
      <li><b>Model threshold at the ADA test's number of tests:</b> ${r57.p >= EBM.thr ? "would be sent for a test" : "would not be sent"}</li>
    </ul>
    <div class="chart short" id="contrib-chart"></div>
    <p class="note">Arm, leg length and household income are not asked: they are set to typical values for your age, sex and size and left out of the chart, as is any blank field${x.filled.length ? ` (filled: ${x.filled.map((k) => PRETTY[k].toLowerCase()).join(", ")})` : ""}.</p>`;
  Plotly.newPlot("contrib-chart", [{
    type: "bar", orientation: "h", x: top.map((c) => c[1]), y: top.map((c) => PRETTY[c[0]] || c[0]),
    marker: { color: top.map((c) => (c[1] > 0 ? "#c98a0b" : "#2f6fde")) }, hovertemplate: "%{y}: %{x:+.2f} log-odds<extra></extra>",
  }], layout({ margin: { l: 150, r: 20, t: 30, b: 40 }, title: { text: "What moves this estimate (log-odds, HbA1c 5.7%+)", font: { size: 13 }, x: 0 },
    xaxis: { zeroline: true, zerolinecolor: "#17243b", gridcolor: "#e6e9ee" } }), CONFIG);
}

function initCalculator(e57, e65, R) {
  EBM.y57 = e57;
  EBM.y65 = e65;
  EBM.thr = R.frozen.thresholds["ebm|y57"].ada;
  const form = $("calc-form");
  const g = document.createElement("div");
  g.className = "row";
  g.innerHTML = '<span>Group, for the screening rules only <span class="optional">optional</span></span><select name="group"><option value="">Prefer not to say</option><option value="white">Non-Hispanic White</option><option value="black">Black</option><option value="hispanic">Hispanic</option><option value="asian">Asian</option><option value="other">Other</option></select>';
  form.appendChild(g);
  form.addEventListener("input", runCalculator);
  runCalculator();
}

async function main() {
  const [R, Y, e57, e65] = await Promise.all(["results.json", "yield_curves.json", "ebm_y57.json", "ebm_y65.json"].map(getJSON));
  if (!R) return;
  renderFlow(R.flow);
  renderValidation(R.validation_mean);
  renderLadder(R.leakage_ladder);
  if (R.main_table && Y) {
    $("results-body").hidden = false;
    renderHero(R, Y);
    renderOps(R);
    renderYield(Y);
    renderMain(R);
    renderDisc(R);
    renderH(R);
    renderCycles(R);
    renderSubgroups(R);
    renderMissed(R);
    renderConclusion(R);
  }
  if (e57 && e65) initCalculator(e57, e65, R);
  const redraw = () => {
    renderLadder(R.leakage_ladder);
    if (R.main_table && Y) {
      renderOps(R);
      ["rate", "dim"].forEach((id) => $(id).dispatchEvent(new Event("input")));
    }
  };
  if (NARROW.addEventListener) NARROW.addEventListener("change", redraw);
}

window.addEventListener("DOMContentLoaded", () => {
  const go = () => (window.Plotly ? main() : setTimeout(go, 50));
  go();
});
