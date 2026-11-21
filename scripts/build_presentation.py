"""Build corrected editable decks, speaker notes and previews from saved results.

No model is loaded or executed. Uses standard-library OOXML, Pillow and matplotlib.
Compact plot data is retained so the deck can be rebuilt on the lab PC.
"""

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import textwrap
import zipfile
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-slide-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/presentation"
DECKS = ROOT / "presentations"
BACKUP = ROOT / "temp/presentation_before_review_2026-11-21"
PREVIEW = ROOT / "temp/presentation_review_previews"
W, H = 1600, 900
BG, PANEL, INK, MUTED = "#0D1424", "#172238", "#F6F8FC", "#B7C5DA"
TEAL, PURPLE, GOLD, BLUE = "#49D6B5", "#BC9DFF", "#F5C56B", "#79BFFF"
COLORS = {"128": "#95A4B8", "256": PURPLE, "patchstg": TEAL}
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data():
    compact = OUT / "slide_data.json"
    saved = read_json(compact) if compact.exists() else {}
    histories, provenance = {}, {}
    for key in ("128", "256", "patchstg"):
        path = ROOT / ("runs/patchstg_sd_seed2023_50ep/report.json" if key == "patchstg" else
                       f"runs/context_lstm_suite_v1/hidden_{key}/report.json")
        if path.exists():
            report = read_json(path)
            history = report["history"] if key == "patchstg" else report["epochs"]
            histories[key] = [{"epoch": e["epoch"], "train_mae": e["training"]["mae"],
                               "validation_mae": e["validation"]["overall_pooled"]["mae"] if key == "patchstg"
                               else e["validation"]["mae"]} for e in history]
            provenance[key] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
        elif key in saved.get("histories", {}):
            histories[key] = saved["histories"][key]
            provenance[key] = saved["history_sources"][key]
        else:
            raise FileNotFoundError(f"Missing saved training history for {key}; no values will be invented.")
        assert [e["epoch"] for e in histories[key]] == list(range(1, 51))
    lstm = read_json(ROOT / "reports/final_comparison/report.json")
    patch = read_json(ROOT / "reports/patchstg_sd_seed2023_test_v1.json")
    comparison = read_json(ROOT / "reports/final_comparison/comparison.json")
    example = saved.get("example")
    page = ROOT / "runs/lstm_patchstg_replay_v1/index.html"
    if page.exists():
        match = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', page.read_text(), re.S)
        data = json.loads(match.group(1))
        frame, sensor = 0, 0  # Predefined earliest origin/first sensor; no performance selection.
        offset = data["frames"][frame]["observation_offset"]
        example = {"sensor": data["sensors"][sensor]["id"], "issued_at": data["frames"][frame]["issued_at"],
                   "history": [r[sensor] for r in data["observations"][offset - 12:offset]],
                   "actual": [r[sensor] for r in data["observations"][offset:offset + 12]],
                   "lstm": data["models"]["lstm"]["predictions"][frame][sensor],
                   "patchstg": data["models"]["patchstg"]["predictions"][frame][sensor],
                   "selection": "first chronological replay origin and first sensor; not error-selected",
                   "source_sha256": digest(page)}
    if example is None:
        raise FileNotFoundError("Saved illustrative forecast data is unavailable.")
    payload = {"histories": histories, "history_sources": provenance, "example": example,
               "lstm_test": lstm["overall"], "patchstg_test": patch["results"]["test"]["overall_pooled"],
               "note": "Existing completed measurements only; no training or new inference."}
    compact.write_text(json.dumps(payload, indent=2) + "\n")
    return payload, lstm, patch, comparison


def style_plots():
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": "#496078", "grid.color": "#344158",
                         "font.size": 20, "axes.titlesize": 23, "axes.labelsize": 21,
                         "legend.fontsize": 17, "legend.facecolor": PANEL,
                         "legend.edgecolor": "#496078", "figure.dpi": 140})


def save_plot(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_figures(data, lstm, patch):
    style_plots()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.1), layout="constrained")
    for key, label in (("128", "LSTM 128"), ("256", "LSTM 256"), ("patchstg", "PatchSTG")):
        history = data["histories"][key]
        for ax, field in zip(axes, ("train_mae", "validation_mae")):
            ax.plot([e["epoch"] for e in history], [e[field] for e in history], label=label,
                    color=COLORS[key], linewidth=2.5, linestyle="--" if key == "128" else "-")
        best = min(history, key=lambda e: e["validation_mae"])
        axes[1].scatter(best["epoch"], best["validation_mae"], s=90, color=COLORS[key], zorder=4)
    axes[0].set(title="Training MAE", xlabel="Epoch", ylabel="MAE")
    axes[1].set(title="Validation MAE", xlabel="Epoch", ylabel="MAE")
    for ax in axes:
        ax.grid(alpha=.4); ax.legend()
    save_plot(fig, "learning_curves_all_models.png")
    # Separate large validation view for questions/appendix.
    fig, ax = plt.subplots(figsize=(13, 5), layout="constrained")
    for key, label in (("128", "LSTM 128"), ("256", "LSTM 256"), ("patchstg", "PatchSTG")):
        h = data["histories"][key]
        ax.plot([e["epoch"] for e in h], [e["validation_mae"] for e in h], label=label,
                color=COLORS[key], linewidth=2.6, linestyle="--" if key == "128" else "-")
        b = min(h, key=lambda e: e["validation_mae"])
        ax.scatter(b["epoch"], b["validation_mae"], color=COLORS[key], s=100)
    ax.set(xlim=(8, 51), ylim=(14.6, 18), xlabel="Epoch (zoom: epochs 8–50)", ylabel="Validation MAE")
    ax.grid(alpha=.4); ax.legend(); save_plot(fig, "validation_zoom_all_models.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.1), layout="constrained")
    reference = ROOT / "reports"
    for name, filename, color in (("Persistence", "persistence_sd_v1.json", "#95A4B8"),
                                   ("Seasonal average", "seasonal_average_sd_v1.json", GOLD)):
        h = read_json(reference / filename)["results"]["test"]["per_horizon"]
        axes[0].plot(range(15, 181, 15), [e["mae"] for e in h], label=name, color=color, linewidth=2.6)
    for name, h, color in (("LSTM 256", lstm["per_horizon"], PURPLE),
                            ("PatchSTG", patch["results"]["test"]["per_horizon"], TEAL)):
        for ax in axes:
            ax.plot(range(15, 181, 15), [e["mae"] for e in h], color=color, label=name, linewidth=2.8)
    axes[0].set_title("All methods")
    axes[1].set(title="Neural-model zoom", ylim=(10, 24))
    for ax in axes:
        ax.set(xlabel="Minutes ahead (assumed)", ylabel="Test MAE", xticks=(15, 60, 120, 180))
        ax.grid(alpha=.4); ax.legend(fontsize=15)
    save_plot(fig, "forecast_horizon_comparison.png")
    fig, ax = plt.subplots(figsize=(13, 4.8), layout="constrained")
    values = [e["mae"] for e in lstm["per_sensor"] if e["mae"] is not None]
    ax.hist(values, bins=30, color=PURPLE, edgecolor=BG)
    ax.set(xlabel="Per-sensor test MAE", ylabel="Number of sensors", title="LSTM-only sensor diagnostic (696 scored sensors)")
    ax.grid(axis="y", alpha=.3); save_plot(fig, "sensor_errors_lstm.png")
    e = data["example"]
    fig, ax = plt.subplots(figsize=(13, 4.8), layout="constrained")
    ax.plot(range(-165, 1, 15), e["history"], color=BLUE, linewidth=2.7, label="Observed history")
    for key, label, color, ls in (("lstm", "LSTM 256", PURPLE, "-"),
                                   ("patchstg", "PatchSTG", TEAL, "-"),
                                   ("actual", "Actual future", GOLD, "--")):
        ax.plot(range(0, 181, 15), [e["history"][-1], *e[key]], label=label, color=color, linewidth=2.7, linestyle=ls)
    ax.axvline(0, color=MUTED, linestyle=":")
    ax.set(xlabel="Minutes from forecast issue", ylabel="Traffic flow (source units)", xticks=(-165, -90, 0, 90, 180))
    ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=17); save_plot(fig, "matched_forecast_example.png")


FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(FONT_DIR / filename), round(size * H / 540))


def wrap(text, width, size, bold=False):
    f = font(size, bold)
    result = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            result.append(""); continue
        line = words.pop(0)
        for word in words:
            if f.getlength(line + " " + word) <= width:
                line += " " + word
            else:
                result.append(line); line = word
        result.append(line)
    return result


class Slide:
    def __init__(self, title, section, notes="", minutes=.8):
        self.title, self.section, self.notes, self.minutes = title, section, notes, minutes
        self.objects = []
        self.text("PRACTICAL MACHINE LEARNING  /  TRAFFIC FORECASTING", 70, 35, 1100, 30, 12, MUTED)
        self.text(title, 70, 92, 1460, 92, 34, INK, True)
        self.rect(70, 192, 1460, 3, TEAL)

    def rect(self, x, y, w, h, fill=PANEL):
        self.objects.append(dict(kind="rect", x=x, y=y, w=w, h=h, fill=fill))

    def text(self, text, x, y, w, h, size=24, color=INK, bold=False):
        lines = wrap(text, w, size, bold)
        needed = len(lines) * size * H / 540 * 1.18
        while needed > h + 2 and size > 18:
            size -= 1
            lines = wrap(text, w, size, bold)
            needed = len(lines) * size * H / 540 * 1.18
        if needed > h + 2:
            raise ValueError(f"Text exceeds its box on {self.title}: {text[:70]!r}; {needed:.0f}>{h}")
        self.objects.append(dict(kind="text", text="\n".join(lines), x=x, y=y, w=w, h=h,
                                 size=size, color=color, bold=bold))

    def image(self, path, x=75, y=230, w=1450, h=515):
        with Image.open(path) as im:
            ratio = min(w / im.width, h / im.height)
            iw, ih = im.width * ratio, im.height * ratio
        self.objects.append(dict(kind="image", path=str(path), x=x + (w-iw)/2, y=y+(h-ih)/2, w=iw, h=ih))

    def line(self, x1, y1, x2, y2, color=MUTED):
        self.objects.append(dict(kind="line", x=x1, y=y1, w=x2-x1, h=y2-y1, color=color))

    def bullets(self, items, x=95, y=250, w=1400, size=25, gap=105):
        for i, item in enumerate(items):
            self.text("• " + item, x, y + i*gap, w, gap-12, size)

    def cards(self, items, y=280, h=310):
        gap = 24; width = (1460 - gap*(len(items)-1))/len(items)
        for i, (heading, body) in enumerate(items):
            x = 70 + i*(width+gap)
            self.rect(x, y, width, h)
            self.text(heading, x+22, y+24, width-44, 80, 23, TEAL, True)
            self.text(body, x+22, y+115, width-44, h-125, 22)

    def table(self, headers, rows, widths, y=245, row_height=88, size=22):
        x=80
        for i, row in enumerate([headers, *rows]):
            xx=x; yy=y+i*row_height
            self.rect(x, yy, sum(widths), row_height-3, "#223650" if i==0 else PANEL)
            for j, value in enumerate(row):
                self.text(str(value), xx+16, yy+16, widths[j]-32, row_height-19,
                          size, TEAL if i==0 else INK, i==0)
                xx+=widths[j]

    def takeaway(self, text):
        self.text(text, 80, 750, 1440, 96, 23, GOLD, True)


def main_slides(data, lstm, patch):
    s=[]
    def add(title, section, notes, minutes=.8):
        a=Slide(title,section,notes,minutes);s.append(a);return a
    a=add("Forecasting traffic across 716 sensors", "Opening",
          "Introduce the practical forecasting question and the course-method LSTM. PatchSTG is integrated; its completed run is reused, not retrained. Present measured comparisons rather than promising a particular architecture should win.", .45)
    a.text("Contextual LSTM versus PatchSTG",90,270,1400,100,36,TEAL,True)
    a.text("SH Emon  |  Practical Machine Learning\nNovember 2026",95,430,1400,160,27)
    a.takeaway("12 observed steps → 12 future steps  |  Processed LargeST-SD")
    a=add("Which system forecasts the next three hours better?", "Question",
          "Traffic can reflect recent trends, recurring calendar patterns and relationships between locations. Compare a contextual temporal LSTM with PatchSTG on matched forecast targets. We do not isolate the causal effect of spatial attention because capacity and optimization also differ.", .9)
    a.cards([("Temporal context", "Recent flow, time of day, weekday and sensor identity"),
             ("Spatial interaction", "Can other sensors' histories improve the complete forecasting system?"),
             ("Shared assessment", "Same history, horizon, chronological targets and zero mask")],h=340)
    a.takeaway("A practical system comparison—not a controlled spatial-attention ablation.")
    a=add("Dataset: processed LargeST-SD, San Diego", "Data",
          "The processed array is treated as SD 2019: 35,040 time positions, 716 sensors, one traffic-flow feature. Fifteen-minute cadence and January 1 midnight origin are assumptions: the NPZ has no timestamps or sensor IDs. Metadata ordering remains assumed. Cite Liu et al., LargeST, NeurIPS 2023.", .85)
    a.table(["Property","Value"],[("Time steps × sensors × feature","35,040 × 716 × 1"),
            ("History → horizon","12 → 12 steps; 3 hours → 3 hours"),
            ("Sampling / origin","Assumed 15 min; 2019-01-01"),
            ("Forecast variable","Traffic flow in original source units")],[670,770],row_height=100)
    a.takeaway("One example contains all sensors; each model uses that information differently.")
    a=add("Chronological targets prevent overlapping-window leakage", "Protocol",
          "Training targets are [0,21024), validation [21024,28032), test [28032,35040). All 12 targets stay inside their split; earlier observations can supply history. Fit one global scaler on training including zeros. Keep zero inputs, exclude zero targets as a benchmark convention, and score in original units.", 1.15)
    a.cards([("Train: 60%","21,024 frames\nFit weights and scale"),("Validate: 20%","7,008 frames\nChoose model/epoch"),("Test: 20%","7,008 frames\nFreeze, then score")],h=285)
    a.text("Training-only normalization. Zero inputs remain; zero targets are masked.",80,635,1440,110,24)
    a.takeaway("Rolling forecasts may use earlier observed history across a split boundary.")
    a=add("Baselines make the learned-model results meaningful", "References",
          "Persistence repeats the latest reading. Seasonal average uses nonzero training means for sensor, weekday and 15-minute slot, with sensor/global fallbacks. MAE is mean absolute error in original units; RMSE emphasizes large errors; MAPE is relative percentage error. All metrics exclude zero targets and pool errors/counts, not unweighted batch means.", .95)
    a.table(["Method","Prediction rule"],[("Persistence","Repeat the latest observation"),
            ("Seasonal average","Training mean: sensor × weekday × slot")],[430,1010],row_height=110)
    a.text("MAE: absolute error   |   RMSE: larger-error penalty\nMAPE: percentage error; zero actuals excluded",95,610,1400,145,25)
    a.takeaway("All results below use pooled original-unit metrics. Lower is better.")
    a=add("Context makes each historical LSTM step 41-dimensional", "LSTM",
          "A learned embedding is a vector selected by an integer ID, not a new traffic measurement. The same sensor embedding is reused over its 12 historical steps. Input calendar is reconstructed for past steps only. The shared LSTM still processes sensor histories independently.", .85)
    a.table(["Historical input","Values"],[("Normalized flow","1"),("Time-slot embedding","16"),
            ("Weekday embedding","8"),("Sensor-ID embedding","16")],[1130,310],row_height=88)
    a.takeaway("1 + 16 + 8 + 16 = 41 values per sensor per input step.")
    a=add("One shared two-layer LSTM predicts all 12 leads", "LSTM",
          "Reorder (B,12,N,41) to (B,N,12,41), then combine batch and sensor axes. First LSTM layer produces a hidden sequence, the second reads it. Take the top layer's final hidden output, apply a linear 256-to-12 head, restore sensor order and inverse-normalize. Fresh states per independent window. All future outputs are direct, not autoregressive.", 1.15)
    a.cards([("Input sequence","12 historical steps\n41 values per step"),("Two LSTM layers","256 hidden units each\nFresh recurrent states"),("Forecast head","Final hidden → 12\nOriginal flow units")],h=340)
    a.takeaway("Shared parameters across sensors; no exchange of neighboring traffic histories.")
    a=add("PatchSTG encodes history, then exchanges spatial information", "PatchSTG",
          "Fang et al., KDD 2025, provides the PatchSTG reference. The integrated shared-protocol adaptation projects 12 steps of traffic/numeric calendar, concatenates learned context to width 192, gathers 16 patches of 64 padded positions, applies five dual-attention blocks, restores 716 originals, and forecasts 12 normalized values. Inverse scaling occurs at loss/evaluation. Coordinates and training-derived similarity prepare patches; duplicates are not extra targets.", 1.25)
    a.cards([("Encode","12-step history\n192 values per sensor"),("Patch + attend","16 × 64 positions\nFive dual-attention blocks"),("Restore + predict","716 original sensors\n12 forecasts each")],h=340)
    a.takeaway("Coordinates form patches; training-only similarity chooses padding positions.")
    a=add("Attention combines weights and Values as separate inputs", "PatchSTG",
          "Queries and Keys produce softmax-scaled scores. Values are a separate projection of representations, not derived from the scores. Multiply the weights matrix by Values. Depth attention uses 64 tokens in one patch. Breadth attention uses 16 tokens at matching positions across patches, attending jointly rather than passing sequentially from patch 0 to patch 15. Depth runs first within each block.", 1.35)
    a.rect(80,250,700,180);a.text("Weights = softmax(QKᵀ / √192)\nDepth shape: 64 × 64",105,280,650,130,25,TEAL,True)
    a.rect(840,250,680,180);a.text("Values: separate learned projection\nDepth shape: 64 × 192",865,280,630,130,25,PURPLE,True)
    a.line(430,440,680,535);a.line(1180,440,920,535)
    a.rect(465,540,670,115);a.text("Weights × Values → 64 × 192",490,565,620,70,25,INK,True)
    a.text("1. Depth: 64 positions within a patch\n2. Breadth: 16 matching positions across patches, jointly",95,675,1400,95,23)
    a=add("Shared task and context; different modeling budgets", "Comparison",
          "Both have historical traffic, time/weekday and sensor identity. LSTM is per-sensor; PatchSTG mixes spatial representations and uses coordinates/similarity. PatchSTG has 2,284,844 parameters versus 848,644. Similar context and shared splits improve comparability but parameter count, optimization and search effort differ. No causal attention-only claim.", .95)
    a.table(["Aspect","Contextual LSTM","PatchSTG"],[("History / horizon","12 / 12","12 / 12"),
            ("Calendar + sensor ID","Yes","Yes"),("Sensor interaction","Independent histories","Spatial attention"),
            ("Parameters","848,644","2,284,844")],[410,515,515],row_height=90,size=21)
    a.takeaway("Compare complete systems; disclose the remaining differences.")
    a=add("Training protocol: 50 epochs, batch 4, seed 2023", "Training",
          "Show the actual settings rather than only the generic learning loop. LSTM Adam has fixed lr .001 without schedule/weight decay/clipping. PatchSTG AdamW starts .002, decays at epoch ends 1/35/40 by .5, weight decay .0001 and gradient clipping 5. Both use original-unit masked MAE and full validation. Same epoch count is not equal compute or search budget.", 1.05)
    a.table(["Setting","Contextual LSTM","PatchSTG"],[("Optimizer","Adam","AdamW"),
            ("Learning rate","0.001 fixed","0.002 + schedule"),("Decay / clipping","None / none","0.0001 / norm 5"),
            ("Model selection","Lowest validation MAE","Lowest validation MAE")],[410,515,515],row_height=90,size=21)
    a.takeaway("Same data passes; different optimizer, schedule, capacity and tuning effort.")
    a=add("Validation selects width 256, epoch 37", "Selection",
          "The two completed LSTM candidates each trained 50 epochs. Width 256 best validation is 15.0656 at epoch 37; width 128 is 15.3385 at epoch 39. PatchSTG selected epoch 50 at 14.9936. Width 384 was deferred after interim validation and before the contextual test pass, revising the search scope. Do not call it a completed three-width search.", .8)
    a.table(["Candidate","Best epoch","Validation MAE"],[("LSTM 128","39","15.3385"),
            ("LSTM 256 — selected","37","15.0656"),("PatchSTG","50","14.9936")],[690,300,450],row_height=105,size=22)
    a.takeaway("Both LSTMs completed 50 epochs. Width 384 is deferred, not silently counted.")
    a=add("Learning curves now include PatchSTG's complete run", "Analysis",
          "All three curves use their real saved 50-epoch reports. Training MAE pools evolving weights, while validation uses completed-epoch frozen weights. The 256 LSTM lowers training error relative to 128, but validation gains are modest. PatchSTG starts higher, then ends with the best recorded validation score. Different optimization prevents attributing trajectories solely to architecture. Dots mark each model's selected epoch.", 1.25)
    a.image(OUT/'learning_curves_all_models.png',y=240,h=485)
    a.takeaway("Training improves steadily; validation fluctuates. Select checkpoints, not final epochs.")
    a=add("Frozen test: identical origins, targets and masks", "Evaluation",
          "Select before scoring. The contextual evaluator rechecked full validation, scored test under eval/no-grad, and verified unchanged model/checkpoint state. Reuse PatchSTG's saved completed test pass with matching hashes, origins and lead-wise target counts; it was not rerun. No significance claim from millions of overlapping targets: these are correlated forecast-target pairs, not independent experiments.", .75)
    a.cards([("Freeze","LSTM epoch 37\nPatchSTG epoch 50"),("Score","6,997 test windows\n58,158,396 valid pairs"),("Verify","Matching source/masks\nNo weight updates")],h=330)
    a.takeaway("PatchSTG scores are reused from its saved run; no retraining or new inference.")
    a=add("PatchSTG records the lowest error on all three metrics", "Results",
          "Read the four-method table. LSTM test MAE 18.1696 versus PatchSTG 17.1943 is a difference of .9753 original flow units. PatchSTG reduces MAE by 5.37% relative to LSTM. Saying LSTM is 5.67% higher uses the other denominator; do not mix them. These are observed single-seed differences, not statistical significance.", 1.15)
    a.table(["Method","MAE ↓","RMSE ↓","MAPE ↓"],[("Persistence","60.78","94.34","41.78%"),
            ("Seasonal average","28.00","53.39","20.00%"),("Contextual LSTM","18.17","31.95","12.06%"),
            ("PatchSTG","17.19","29.14","11.17%")],[650,260,260,270],row_height=89,size=22)
    a.takeaway("PatchSTG: 0.98 flow units lower MAE; 5.37% reduction relative to LSTM.")
    a=add("Longer leads hurt persistence; learned forecasts stay stronger", "Analysis",
          "The left plot includes all methods. The right deliberately zooms to the two neural curves with a clearly labeled different y range, so their modest gap is visible. At 15 min: LSTM12.62, Patch11.81; at180:22.11 and20.84. PatchSTG is lower at every recorded lead. Absolute gaps grow .81 to1.27, but relative gains do not monotonically establish increasing spatial benefit. Seasonal averages are almost flat; persistence grows sharply.", 1.25)
    a.image(OUT/'forecast_horizon_comparison.png',y=235,h=500)
    a.takeaway("Neural-model zoom makes the gap visible without hiding the baseline context.")
    a=add("Competitive LSTM accuracy with 37.1% of the parameters", "Analysis",
          "Parameter count is a capacity/storage measure, not an inference-speed benchmark. Selected LSTM848644 is37.1% of Patch2284844, while its error is5.67% higher relative to PatchSTG. This is an informative system trade-off, not proof of efficiency superiority or that parameters alone explain accuracy. We did not control timings.", .85)
    a.cards([("LSTM 256","848,644 parameters\nTest MAE 18.17"),("PatchSTG","2,284,844 parameters\nTest MAE 17.19"),("Trade-off","37.1% of parameters\n5.67% higher MAE")],h=330)
    a.takeaway("Fewer parameters ≠ demonstrated faster inference. Runtime was not controlled.")
    a=add("Sensor averages hide a long tail of hard cases", "Analysis",
          "The histogram is LSTM-only; saved PatchSTG reports lack paired per-sensor predictions, so no comparative histogram is invented. 696 sensors have at least one eligible target,20none. A few large errors create a right tail. This motivates investigation but does not prove broken sensors or spatial causes. Overall pooled MAE weights forecast pairs, unlike equal weighting of these sensor means.", 1.0)
    a.image(OUT/'sensor_errors_lstm.png',y=230,h=500)
    a.takeaway("LSTM-only diagnostic: 696 scored sensors; 20 have no eligible test targets.")
    a=add("A matched forecast connects the numbers to the task", "Demonstration",
          f"This example is sensor {data['example']['sensor']} at the first chronological replay origin, not selected for favorable errors. Show past observations, both forecasts and actual future. The gold future was not an input. Values are cached display-rounded outputs. If time allows, open runs/lstm_patchstg_replay_v1/index.html and change sensor/lead once; otherwise this figure is the offline demo. Do not tune using these test examples.", 1.1)
    a.image(OUT/'matched_forecast_example.png',y=235,h=500)
    a.takeaway("First replay origin + first sensor, fixed by position—not selected by forecast quality.")
    a=add("What the experiment does—and does not—establish", "Limitations",
          "Be explicit: one seed and one network; capacity, spatial information, optimization and tuning history differ. The same test period had already been observed before the contextual redesign. Timestamp/metadata pairing/zero meanings remain assumptions. Do not attribute gains solely to attention or claim significance/scalability. New seeds and controlled ablations are future work, not completed experiments.", 1.0)
    a.bullets(["One recorded seed and one network; no significance claim.",
               "Capacity, spatial inputs, optimizer and search budgets differ.",
               "Test period was observed before the contextual redesign.",
               "Calendar, sensor pairing and zero semantics remain assumptions."],gap=116,size=24)
    a.takeaway("Next: multiple seeds, controlled context/spatial ablations and new evaluation periods.")
    a=add("Contextual LSTM is competitive; PatchSTG is more accurate here", "Conclusion",
          "Summarize three findings in your own words. Both learned models beat simple baselines. PatchSTG is the lowest recorded error. Contextual LSTM provides competitive accuracy at lower parameter count; we cannot claim speed superiority. Calendar/capacity were changed together from the small model, so no isolated causal attribution. Invite questions about data leakage, state/shape and attention.", .7)
    a.bullets(["Both learned models outperform persistence and seasonal average.",
               "PatchSTG MAE 17.19 versus contextual LSTM 18.17.",
               "LSTM uses 37.1% of PatchSTG's parameters."],gap=130,size=27)
    a.takeaway("An observed system-level result—not proof that spatial attention alone caused the gain.")
    a=add("Sources, implementation and reproducible evidence", "References",
          "PatchSTG: Yuchen Fang et al., Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective, accepted SIGKDD2025, arXiv2412.09972v2. LargeST: Xu Liu et al., LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting, NeurIPS2023 datasets/benchmarks. Project uses adapted processed SD and a code-aligned PatchSTG implementation; see attribution, pinned source commit and method deviations. URLs are written on the slide for the exported PDF too.", .35)
    a.text("Fang et al.  |  KDD 2025\nEfficient Large-Scale Traffic Forecasting with Transformers:\nA Spatial Data Management Perspective\narxiv.org/abs/2412.09972v2",90,235,1420,205,21)
    a.text("Liu et al.  |  NeurIPS 2023\nLargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting\narxiv.org/abs/2306.08259",90,475,1420,165,21)
    a.text("Project: github.com/emon109848/PML_PatchSTG_VS_LSTM\nSaved evidence: reports/final_comparison/PROJECT_REPORT.md",90,675,1420,105,20,TEAL)
    a=add("Questions", "Discussion", "Thank the audience. Use the detailed deck's appendix to answer tensor-shape, masking, gate, map, optimizer or numerical-check questions. Do not present backup material unless needed.", 0)
    a.text("Thank you",100,300,1400,125,52,TEAL,True)
    a.text("Traffic forecasting with contextual LSTM and PatchSTG",105,490,1390,125,29)
    return s


def appendix_slides():
    cache = OUT / "appendix_content.json"
    if not cache.exists() or (BACKUP / "presentation_detailed.pptx").exists():
        old = BACKUP / "presentation_detailed.pptx"
        contents=[]
        with zipfile.ZipFile(old) as z:
            for number in range(35,56):
                tree=ET.fromstring(z.read(f"ppt/slides/slide{number}.xml"))
                shapes=[]
                for shape in tree.findall('.//p:sp',NS):
                    lines=[''.join('\n' if n.tag == '{'+NS['a']+'}br' else (n.text or '')
                                   for n in p.iter() if n.tag in ('{'+NS['a']+'}t','{'+NS['a']+'}br'))
                           for p in shape.findall('.//a:p',NS)]
                    value='\n'.join(lines).strip()
                    if value:shapes.append(value)
                notes=ET.fromstring(z.read(f"ppt/notesSlides/notesSlide{number}.xml"))
                contents.append({"title":shapes[1],"items":[v for v in shapes[4:] if v!='›' and not v.startswith('Detailed explanation')],
                                 "notes":' '.join(n.text or '' for n in notes.findall('.//a:t',NS))})
        cache.write_text(json.dumps(contents,indent=2)+'\n')
    slides=[]
    for row in read_json(cache):
        a=Slide(row['title'],'Backup',row['notes'],0)
        values=row['items']
        pairs=[(values[i],values[i+1]) for i in range(0,len(values)-1,2)]
        a.cards(pairs[:3],y=285,h=365)
        a.takeaway('Backup explanation · use only when this detail is relevant to a question.')
        slides.append(a)
    # Replace ambiguous sequential/shape presentation with exact two-input wording.
    for a in slides:
        if a.title=='Where the Value vectors are added':
            a.notes+=' Values are independently projected from input representations, not produced by attention scores.'
    return slides


def render(slide,number,total,path):
    image=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(image)
    for item in slide.objects:
        x,y,w,h=[item[k] for k in ('x','y','w','h')]
        if item['kind']=='rect':d.rectangle((x,y,x+w,y+h),fill=item['fill'])
        elif item['kind']=='line':d.line((x,y,x+w,y+h),fill=item['color'],width=3)
        elif item['kind']=='image':
            with Image.open(item['path']) as source:image.paste(source.convert('RGB').resize((round(w),round(h)),Image.Resampling.LANCZOS),(round(x),round(y)))
        else:
            for i,line in enumerate(item['text'].split('\n')):
                d.text((x,y+i*item['size']*H/540*1.18),line,font=font(item['size'],item['bold']),fill=item['color'],anchor='lt')
    d.text((70,860),slide.section.upper(),font=font(12),fill=MUTED,anchor='lt')
    d.text((1450,860),f'{number:02d} / {total:02d}',font=font(12),fill=MUTED,anchor='lt')
    image.save(path)
    return image


def sub(parent,prefix,tag,**attrs):
    return ET.SubElement(parent,'{'+NS[prefix]+'}'+tag,{k:str(v) for k,v in attrs.items()})


def color(parent,value):
    fill=sub(parent,'a','solidFill');sub(fill,'a','srgbClr',val=value.lstrip('#'))


def text_shape(tree,item,index,placeholder=False):
    sp=sub(tree,'p','sp');nv=sub(sp,'p','nvSpPr');sub(nv,'p','cNvPr',id=index,name=f'Text {index}')
    sub(nv,'p','cNvSpPr',txBox=1);pr=sub(nv,'p','nvPr')
    if placeholder:sub(pr,'p','ph',type='body',idx=1)
    properties=sub(sp,'p','spPr');transform=sub(properties,'a','xfrm')
    sub(transform,'a','off',x=round(item['x']*7620),y=round(item['y']*7620))
    sub(transform,'a','ext',cx=round(item['w']*7620),cy=round(item['h']*7620))
    geom=sub(properties,'a','prstGeom',prst='rect');sub(geom,'a','avLst');sub(properties,'a','noFill')
    line=sub(properties,'a','ln');sub(line,'a','noFill')
    body=sub(sp,'p','txBody');sub(body,'a','bodyPr',wrap='square',lIns=0,rIns=0,tIns=0,bIns=0,anchor='t')
    sub(body,'a','lstStyle')
    for line in item['text'].split('\n'):
        paragraph=sub(body,'a','p');pp=sub(paragraph,'a','pPr')
        spacing=sub(pp,'a','lnSpc');sub(spacing,'a','spcPts',val=round(item['size']*118))
        run=sub(paragraph,'a','r');rp=sub(run,'a','rPr',lang='en-US',sz=round(item['size']*100),b=int(item.get('bold',False)))
        color(rp,item['color']);sub(rp,'a','latin',typeface='Arial');sub(rp,'a','ea',typeface='Arial')
        sub(run,'a','t').text=line
    return sp


def tree_root(kind='sld'):
    root=ET.Element('{'+NS['p']+'}'+kind)
    cs=sub(root,'p','cSld');bg=sub(cs,'p','bg');props=sub(bg,'p','bgPr');color(props,BG)
    tree=sub(cs,'p','spTree');nv=sub(tree,'p','nvGrpSpPr');sub(nv,'p','cNvPr',id=1,name='')
    sub(nv,'p','cNvGrpSpPr');sub(nv,'p','nvPr');sub(tree,'p','grpSpPr')
    return root,tree


def xml_bytes(root):
    return ET.tostring(root,encoding='utf-8',xml_declaration=True)


def relationships(entries):
    root=ET.Element('Relationships',xmlns=REL)
    for ident,kind,target in entries:
        ET.SubElement(root,'Relationship',Id=ident,Type=NS['r']+'/'+kind,Target=target)
    return xml_bytes(root)


def export_pptx(slides,path,base):
    with zipfile.ZipFile(base) as archive:
        files={n:archive.read(n) for n in archive.namelist() if not n.startswith(('ppt/slides/','ppt/notesSlides/','ppt/media/'))}
    pres=ET.fromstring(files['ppt/presentation.xml']);lst=pres.find('p:sldIdLst',NS);lst.clear()
    size=pres.find('p:sldSz',NS);size.set('cx','12192000');size.set('cy','6858000');size.set('type','screen16x9')
    relroot=ET.fromstring(files['ppt/_rels/presentation.xml.rels'])
    for child in list(relroot):
        if child.attrib['Type'].endswith('/slide'):relroot.remove(child)
    media={}
    for i,slide in enumerate(slides,1):
        sid=f'rId{1000+i}';sub(lst,'p','sldId',id=255+i,**{'{'+NS['r']+'}id':sid})
        ET.SubElement(relroot,'{'+REL+'}Relationship',Id=sid,Type=NS['r']+'/slide',Target=f'slides/slide{i}.xml')
        page,tree=tree_root();entries=[('rId1','slideLayout','../slideLayouts/slideLayout7.xml'),('rId2','notesSlide',f'../notesSlides/notesSlide{i}.xml')]
        objects=list(slide.objects)+[dict(kind='text',text=slide.section.upper(),x=70,y=860,w=1100,h=30,size=12,color=MUTED,bold=False),
                                     dict(kind='text',text=f'{i:02d} / {len(slides):02d}',x=1450,y=860,w=100,h=30,size=12,color=MUTED,bold=False)]
        for index,item in enumerate(objects,2):
            if item['kind']=='text':text_shape(tree,item,index)
            elif item['kind']=='image':
                filename=f'image{len(media)+1}.png' if item['path'] not in media else media[item['path']]
                media[item['path']]=filename;rid=f'rIdImage{index}'
                entries.append((rid,'image','../media/'+filename))
                pic=sub(tree,'p','pic');nv=sub(pic,'p','nvPicPr');sub(nv,'p','cNvPr',id=index,name=Path(item['path']).stem,descr=slide.title)
                sub(nv,'p','cNvPicPr');sub(nv,'p','nvPr');fill=sub(pic,'p','blipFill');sub(fill,'a','blip',**{'{'+NS['r']+'}embed':rid})
                stretch=sub(fill,'a','stretch');sub(stretch,'a','fillRect');pr=sub(pic,'p','spPr');x=sub(pr,'a','xfrm')
                sub(x,'a','off',x=round(item['x']*7620),y=round(item['y']*7620));sub(x,'a','ext',cx=round(item['w']*7620),cy=round(item['h']*7620))
                g=sub(pr,'a','prstGeom',prst='rect');sub(g,'a','avLst')
            else:
                sp=sub(tree,'p','sp');nv=sub(sp,'p','nvSpPr');sub(nv,'p','cNvPr',id=index,name=f'Shape {index}');sub(nv,'p','cNvSpPr');sub(nv,'p','nvPr')
                pr=sub(sp,'p','spPr');x=sub(pr,'a','xfrm',flipH=int(item['w']<0),flipV=int(item['h']<0))
                sub(x,'a','off',x=round(min(item['x'],item['x']+item['w'])*7620),y=round(min(item['y'],item['y']+item['h'])*7620))
                sub(x,'a','ext',cx=round(abs(item['w'])*7620),cy=max(1,round(abs(item['h'])*7620)))
                g=sub(pr,'a','prstGeom',prst='line' if item['kind']=='line' else 'rect');sub(g,'a','avLst')
                if item['kind']=='rect':color(pr,item['fill']);ln=sub(pr,'a','ln');sub(ln,'a','noFill')
                else:ln=sub(pr,'a','ln',w=22000);color(ln,item['color'])
        override=sub(page,'p','clrMapOvr');sub(override,'a','masterClrMapping')
        files[f'ppt/slides/slide{i}.xml']=xml_bytes(page)
        files[f'ppt/slides/_rels/slide{i}.xml.rels']=relationships(entries)
        notes,nt=tree_root('notes');text_shape(nt,dict(text=slide.notes,x=80,y=120,w=1400,h=630,size=12,color=INK,bold=False),2,True)
        files[f'ppt/notesSlides/notesSlide{i}.xml']=xml_bytes(notes)
        files[f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels']=relationships([('rId1','notesMaster','../notesMasters/notesMaster1.xml'),('rId2','slide',f'../slides/slide{i}.xml')])
    files['ppt/presentation.xml']=xml_bytes(pres);files['ppt/_rels/presentation.xml.rels']=xml_bytes(relroot)
    for source,name in media.items():files['ppt/media/'+name]=Path(source).read_bytes()
    ct=ET.fromstring(files['[Content_Types].xml']);ctns='http://schemas.openxmlformats.org/package/2006/content-types'
    for child in list(ct):
        if child.attrib.get('PartName','').startswith(('/ppt/slides/','/ppt/notesSlides/')):ct.remove(child)
    for i in range(1,len(slides)+1):
        for folder,stem,kind in (('slides','slide','slide'),('notesSlides','notesSlide','notesSlide')):
            ET.SubElement(ct,'{'+ctns+'}Override',PartName=f'/ppt/{folder}/{stem}{i}.xml',ContentType=f'application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml')
    files['[Content_Types].xml']=xml_bytes(ct)
    app=ET.Element('Properties',xmlns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties')
    for tag,value in (('Application','Microsoft Office PowerPoint'),('PresentationFormat','On-screen Show (16:9)'),('Slides',len(slides)),('Notes',len(slides)),('HiddenSlides',0)):
        ET.SubElement(app,tag).text=str(value)
    files['docProps/app.xml']=xml_bytes(app)
    core=ET.fromstring(files['docProps/core.xml'])
    for node in core:
        if node.tag.endswith('}title'):node.text='Traffic forecasting with contextual LSTM and PatchSTG'
        if node.tag.endswith('}creator') or node.tag.endswith('}lastModifiedBy'):node.text='SH Emon'
    files['docProps/core.xml']=xml_bytes(core)
    with Image.open(PREVIEW/'main/slide01.png') as image:
        thumb=io.BytesIO();image.resize((320,180)).convert('RGB').save(thumb,format='JPEG');files['docProps/thumbnail.jpeg']=thumb.getvalue()
    temporary=path.with_suffix('.pptx.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for name,contents in files.items():z.writestr(name,contents)
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True);PREVIEW.mkdir(parents=True,exist_ok=True);DECKS.mkdir(exist_ok=True)
    data,lstm,patch,comparison=load_data();make_figures(data,lstm,patch)
    main=main_slides(data,lstm,patch);detailed=main+appendix_slides()
    previews=[]
    for label,slides in (('main',main),('detailed',detailed)):
        target=PREVIEW/label;target.mkdir(exist_ok=True)
        for i,slide in enumerate(slides,1):
            for obj in slide.objects:
                if obj['kind']!='line' and (obj['x']<0 or obj['y']<0 or obj['x']+obj['w']>W+1 or obj['y']+obj['h']>H+1):
                    raise ValueError(f'Slide object outside canvas: {slide.title}')
            image=render(slide,i,len(slides),target/f'slide{i:02}.png')
            if label=='main':previews.append(image)
    base=BACKUP/'presentation.pptx'
    if not base.exists():base=DECKS/'presentation.pptx'
    # Read template into memory before replacing either deck.
    template_copy=PREVIEW/'base_template.pptx'
    template_copy.write_bytes(base.read_bytes())
    export_pptx(main,DECKS/'presentation.pptx',template_copy)
    export_pptx(detailed,DECKS/'presentation_detailed.pptx',template_copy)
    previews[0].save(DECKS/'presentation.pdf',save_all=True,append_images=previews[1:],resolution=120)
    sheet=Image.new('RGB',(1280,math.ceil(len(main)/4)*205),BG)
    draw=ImageDraw.Draw(sheet)
    for i,image in enumerate(previews):
        sheet.paste(image.resize((320,180)),((i%4)*320,(i//4)*205))
        draw.text(((i%4)*320+8,(i//4)*205+181),f'{i+1}. {main[i].section}',fill=INK)
    sheet.save(PREVIEW/'contact_sheet.png')
    parts=['# Traffic forecasting presentation — corrected version\n',
           f'{len(main)} main slides for approximately {sum(x.minutes for x in main):.1f} minutes, excluding questions.\n',
           'The detailed deck appends 21 optional lesson/backup slides. Original decks are preserved under temp/presentation_before_review_2026-11-21.\n',
           'All curves use saved measurements. PatchSTG was not retrained or reevaluated. Compact slide data under reports/presentation supports rebuilding on another PC.\n']
    for i,slide in enumerate(detailed,1):
        parts += [f'\n## {"Slide" if i<=len(main) else "Backup"} {i} — {slide.title}\n',f'**Section:** {slide.section} · **Suggested minutes:** {slide.minutes}\n',
                  '**Slide content**\n']
        for obj in slide.objects[3:]:
            if obj['kind']=='text':parts.append('- '+obj['text'].replace('\n',' / ')+'\n')
            elif obj['kind']=='image':parts.append(f'- Figure: [{Path(obj["path"]).name}](../reports/presentation/{Path(obj["path"]).name})\n')
        parts+=['\n**Speaker notes**\n\n'+slide.notes+'\n']
    (DECKS/'presentation.md').write_text('\n'.join(parts))
    manifest={'main_slides':len(main),'detailed_slides':len(detailed),'backup_slides':len(detailed)-len(main),
              'estimated_speaking_minutes':sum(s.minutes for s in main),'curves':['LSTM 128','LSTM 256','PatchSTG'],
              'data_sources':data['history_sources'],'models_executed':False,'checkpoints_modified':False,
              'original_backup':str(BACKUP.relative_to(ROOT)),
              'rebuild':'python scripts/build_presentation.py',
              'preview_note':'Source-layout renders with DejaVu Sans; editable PowerPoint text uses Arial. Not an actual PowerPoint engine render.',
              'outputs':{"presentations/"+name:digest(DECKS/name) for name in ('presentation.pptx','presentation_detailed.pptx','presentation.pdf','presentation.md')}}
    (OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
