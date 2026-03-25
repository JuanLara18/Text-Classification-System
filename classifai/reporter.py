"""
Rich interactive HTML reporter for classifai results.

Generates a single self-contained HTML dashboard with:
- Dataset overview (row count, text length distribution)
- Per-perspective interactive Plotly charts
  · LLM: bar + donut, top keywords, sample texts, balance metrics
  · Clustering: scatter (UMAP/PCA), cluster sizes, top keywords
- Cross-perspective heatmap (category overlap)
- Ground-truth comparison: confusion matrix + precision/recall/F1
  (activated automatically when a `true_category`-style column is present
   and passed as ground_truth_col)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

# ── Plotly color palette ───────────────────────────────────────────────────────
_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


def _palette(n: int) -> List[str]:
    return (_PALETTE * ((n // len(_PALETTE)) + 1))[:n]


# ── Top keywords via TF-IDF ────────────────────────────────────────────────────

def _top_keywords(texts: pd.Series, labels: pd.Series, n_terms: int = 8) -> Dict[str, List[str]]:
    """Return {label: [top_term, ...]} using per-class TF-IDF."""
    result: Dict[str, List[str]] = {}
    try:
        vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform(texts.fillna("").astype(str))
        terms = vec.get_feature_names_out()
        for label in labels.unique():
            mask = (labels == label).values
            if mask.sum() == 0:
                continue
            scores = X[mask].mean(axis=0).A1
            top = terms[scores.argsort()[::-1][:n_terms]].tolist()
            result[label] = top
    except Exception:
        pass
    return result


# ── 2-D projection for clustering scatter ─────────────────────────────────────

def _project_2d(texts: pd.Series, labels: pd.Series, max_pts: int = 2000):
    """Return (x, y, label_subset) using UMAP or PCA fallback."""
    sample_size = min(len(texts), max_pts)
    idx = np.random.RandomState(42).choice(len(texts), sample_size, replace=False)
    t_sub = texts.iloc[idx].fillna("").astype(str)
    l_sub = labels.iloc[idx]

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(t_sub.tolist(), show_progress_bar=False)
    except ImportError:
        from sklearn.feature_extraction.text import TfidfVectorizer
        emb = TfidfVectorizer(max_features=300).fit_transform(t_sub).toarray()
    emb = normalize(emb.astype(float))

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, verbose=False)
        coords = reducer.fit_transform(emb)
    except ImportError:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(emb)

    return coords[:, 0], coords[:, 1], l_sub


# ── Plotly chart builders (return HTML strings) ───────────────────────────────

def _bar_chart(counts: pd.Series, title: str) -> str:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        labels = counts.index.astype(str).tolist()
        values = counts.values.tolist()
        total = sum(values)
        pcts = [f"{v/total*100:.1f}%" for v in values]
        colors = _palette(len(labels))
        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=pcts, textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<br>Share: %{text}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            margin=dict(t=50, b=60, l=40, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0"),
            height=380,
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})
    except ImportError:
        return "<p><em>plotly not installed — install with pip install plotly</em></p>"


def _donut_chart(counts: pd.Series, title: str) -> str:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        labels = counts.index.astype(str).tolist()
        values = counts.values.tolist()
        colors = _palette(len(labels))
        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.45, marker_colors=colors,
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value} rows (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=True, height=380,
            paper_bgcolor="white",
        )
        return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": False})
    except ImportError:
        return ""


def _scatter_chart(x, y, labels, title: str) -> str:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        unique = sorted(set(labels))
        colors = _palette(len(unique))
        color_map = dict(zip(unique, colors))
        traces = []
        for lbl in unique:
            mask = [l == lbl for l in labels]
            xi = [x[i] for i, m in enumerate(mask) if m]
            yi = [y[i] for i, m in enumerate(mask) if m]
            traces.append(go.Scatter(
                x=xi, y=yi, mode="markers",
                name=str(lbl),
                marker=dict(color=color_map[lbl], size=5, opacity=0.7),
                hovertemplate=f"<b>{lbl}</b><extra></extra>",
            ))
        fig = go.Figure(traces)
        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
            yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=50, b=40, l=40, r=20),
            height=460,
        )
        return pio.to_html(fig, full_html=False,
                           include_plotlyjs=False,
                           config={"displayModeBar": False})
    except ImportError:
        return ""


def _heatmap_chart(matrix: pd.DataFrame, title: str) -> str:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        fig = go.Figure(go.Heatmap(
            z=matrix.values,
            x=matrix.columns.astype(str).tolist(),
            y=matrix.index.astype(str).tolist(),
            colorscale="Blues",
            hovertemplate="<b>%{y} → %{x}</b><br>%{z} rows<extra></extra>",
            text=matrix.values,
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=15)),
            margin=dict(t=50, b=80, l=120, r=20),
            paper_bgcolor="white",
            height=max(300, 60 * len(matrix)),
        )
        return pio.to_html(fig, full_html=False,
                           include_plotlyjs=False,
                           config={"displayModeBar": False})
    except ImportError:
        return ""


def _text_length_histogram(texts: pd.Series) -> str:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        lengths = texts.fillna("").astype(str).str.len()
        fig = go.Figure(go.Histogram(
            x=lengths, nbinsx=40,
            marker_color="#4E79A7", opacity=0.8,
            hovertemplate="Length %{x}: %{y} texts<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Text length distribution (characters)", font=dict(size=14)),
            xaxis_title="Characters", yaxis_title="Count",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0"),
            margin=dict(t=50, b=40, l=50, r=20),
            height=300,
        )
        return pio.to_html(fig, full_html=False,
                           include_plotlyjs=False,
                           config={"displayModeBar": False})
    except ImportError:
        return ""


def _confusion_matrix_chart(cm: pd.DataFrame) -> str:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        # Normalize per row (recall perspective)
        cm_norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0).round(2)
        fig = go.Figure(go.Heatmap(
            z=cm_norm.values,
            x=[f"Pred: {c}" for c in cm_norm.columns.astype(str)],
            y=[f"True: {r}" for r in cm_norm.index.astype(str)],
            colorscale="RdYlGn", zmin=0, zmax=1,
            text=cm_norm.values,
            texttemplate="%{text:.0%}",
            hovertemplate="<b>%{y}</b><br><b>%{x}</b><br>Recall: %{z:.1%}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Confusion matrix (row-normalized recall)", font=dict(size=15)),
            margin=dict(t=60, b=100, l=140, r=20),
            paper_bgcolor="white",
            height=max(350, 55 * len(cm_norm)),
        )
        return pio.to_html(fig, full_html=False,
                           include_plotlyjs=False,
                           config={"displayModeBar": False})
    except ImportError:
        return ""


# ── HTML building blocks ──────────────────────────────────────────────────────

_CSS = """
<style>
  :root { --primary: #2c3e6b; --accent: #4E79A7; --bg: #f5f6fa; --card: #fff; }
  * { box-sizing: border-box; }
  body { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; background: var(--bg);
         color: #222; margin: 0; padding: 0; }
  header { background: linear-gradient(135deg, #2c3e6b 0%, #3a5ba0 100%);
           color: #fff; padding: 28px 40px 22px; }
  header h1 { margin: 0 0 4px; font-size: 1.9em; letter-spacing: -0.5px; }
  header p  { margin: 0; opacity: 0.75; font-size: 0.9em; }
  .container { max-width: 1200px; margin: 0 auto; padding: 28px 24px; }
  h2 { font-size: 1.25em; color: var(--primary); margin: 0 0 14px;
       border-left: 4px solid var(--accent); padding-left: 10px; }
  h3 { font-size: 1em; color: #444; margin: 0 0 8px; }
  .card { background: var(--card); border-radius: 10px;
          box-shadow: 0 2px 8px rgba(0,0,0,.07); padding: 22px 24px; margin-bottom: 22px; }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 14px; }
  .metric { background: var(--card); border-radius: 8px; padding: 16px 18px;
            box-shadow: 0 2px 6px rgba(0,0,0,.06); text-align: center; }
  .metric .value { font-size: 2em; font-weight: 700; color: var(--accent); line-height: 1.1; }
  .metric .label { font-size: 0.78em; color: #777; margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media (max-width: 780px) { .chart-grid { grid-template-columns: 1fr; } }
  .chart-box { background: var(--card); border-radius: 10px;
               box-shadow: 0 2px 8px rgba(0,0,0,.07); padding: 16px; }
  .chart-full { background: var(--card); border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,.07); padding: 16px; margin-bottom: 18px; }
  .keyword-cloud { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
  .kw { background: #eef3fb; color: #2c3e6b; border-radius: 20px;
        padding: 4px 12px; font-size: 0.82em; }
  table { width: 100%; border-collapse: collapse; font-size: 0.87em; }
  th { background: #f0f4fa; color: #333; padding: 9px 12px; text-align: left;
       font-weight: 600; border-bottom: 2px solid #dde4f0; }
  td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
  tr:hover td { background: #fafbff; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
           font-size: 0.78em; font-weight: 600; color: #fff; }
  .tag-section { background: #f7f9ff; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }
  .divider { height: 1px; background: #e8edf5; margin: 28px 0; }
  .toc a { color: var(--accent); text-decoration: none; font-size: 0.92em; }
  .toc a:hover { text-decoration: underline; }
  .toc li { margin: 5px 0; }
  .ok   { color: #27ae60; } .warn { color: #e67e22; } .bad { color: #e74c3c; }
</style>
"""


def _metric(value, label, cls=""):
    return f'<div class="metric"><div class="value {cls}">{value}</div><div class="label">{label}</div></div>'


def _sample_table(df: pd.DataFrame, label_col: str, text_col: str, n: int = 3) -> str:
    rows_html = ""
    for cat, group in df.groupby(label_col):
        samples = group[text_col].dropna().astype(str)
        samples = samples[samples.str.strip() != ""].head(n)
        color = _palette(1)[0]
        for i, txt in enumerate(samples):
            short = txt[:220] + ("…" if len(txt) > 220 else "")
            cat_cell = f'<span class="badge" style="background:{color}">{cat}</span>' if i == 0 else ""
            rows_html += f"<tr><td>{cat_cell}</td><td>{short}</td></tr>"
        color = _palette(df[label_col].nunique())[list(df[label_col].unique()).index(cat) % 20]
    return f"""
<table>
  <thead><tr><th style="width:160px">Category</th><th>Sample text</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""


def _keywords_html(kw_dict: Dict[str, List[str]], colors: List[str]) -> str:
    html = ""
    for i, (label, terms) in enumerate(kw_dict.items()):
        color = colors[i % len(colors)]
        badges = " ".join(f'<span class="kw" style="background:#eef3fb;border:1px solid {color};color:{color}">{t}</span>' for t in terms)
        html += f'<div style="margin-bottom:10px"><b style="color:{color}">{label}</b><br>{badges}</div>'
    return html


# ── Main reporter class ────────────────────────────────────────────────────────

class HTMLReporter:
    """
    Generate a rich self-contained HTML report from pipeline results.

    Parameters
    ----------
    df : pd.DataFrame
        The classified dataset (all rows, including result columns).
    perspectives_config : dict
        The `clustering_perspectives` dict from the YAML config.
    results_dir : str
        Directory where the HTML file will be saved.
    text_columns : list of str, optional
        Column(s) that contain the original text (for length histogram and samples).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        perspectives_config: dict,
        results_dir: str,
        text_columns: Optional[List[str]] = None,
    ) -> None:
        self.df = df
        self.persp = perspectives_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.text_columns = text_columns or []

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self, ground_truth_col: Optional[str] = None) -> str:
        """
        Build and save the HTML report.

        Parameters
        ----------
        ground_truth_col : str, optional
            Name of a column with true labels. When present, a confusion matrix
            and classification report are added for each matching perspective.

        Returns
        -------
        str
            Absolute path to the saved HTML file.
        """
        sections: List[str] = []

        # ── 1. Dataset overview ────────────────────────────────────────────────
        sections.append(self._section_overview())

        # ── 2. Per-perspective sections ────────────────────────────────────────
        for name, cfg in self.persp.items():
            ptype = cfg.get("type", "clustering")
            out_col = cfg.get("output_column", name)
            if out_col not in self.df.columns:
                continue
            labels = self.df[out_col].fillna("Unknown").astype(str)

            if ptype == "openai_classification":
                sections.append(self._section_llm(name, cfg, labels, ground_truth_col))
            else:
                sections.append(self._section_clustering(name, cfg, labels))

        # ── 3. Cross-perspective ───────────────────────────────────────────────
        out_cols = {n: cfg.get("output_column", n) for n, cfg in self.persp.items()
                    if cfg.get("output_column", n) in self.df.columns}
        if len(out_cols) >= 2:
            sections.append(self._section_cross(out_cols))

        # ── 4. Assemble HTML ───────────────────────────────────────────────────
        toc_items = "".join(
            f'<li><a href="#{n}">{n}</a></li>' for n in self.persp
        )
        toc = f'<ul class="toc">{toc_items}</ul>'

        generated = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>classifai — results report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
{_CSS}
</head>
<body>
<header>
  <h1>classifai &nbsp;·&nbsp; Results Report</h1>
  <p>Generated {generated} &nbsp;·&nbsp; {len(self.df):,} rows &nbsp;·&nbsp; {len(self.persp)} perspective(s)</p>
</header>
<div class="container">
  <div class="card" style="margin-bottom:22px">
    <h2 style="border:none;padding:0;margin-bottom:8px">Perspectives in this report</h2>
    {toc}
  </div>
  {"".join(sections)}
</div>
</body>
</html>"""

        out_path = self.results_dir / "report.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved → {out_path}")
        return str(out_path)

    # ── Section builders ───────────────────────────────────────────────────────

    def _section_overview(self) -> str:
        df = self.df
        n_rows = len(df)
        n_cols = len(df.columns)

        # Text stats
        if self.text_columns:
            combined = df[self.text_columns[0]].fillna("").astype(str)
            avg_len = int(combined.str.len().mean())
            median_len = int(combined.str.len().median())
            hist_html = _text_length_histogram(combined)
        else:
            avg_len = median_len = 0
            hist_html = ""

        metrics = "".join([
            _metric(f"{n_rows:,}", "Total rows"),
            _metric(f"{n_cols}", "Columns"),
            _metric(f"{len(self.persp)}", "Perspectives"),
            _metric(f"{avg_len:,}", "Avg text length"),
            _metric(f"{median_len:,}", "Median text length"),
        ])

        return f"""
<div class="card" id="_overview">
  <h2>Dataset overview</h2>
  <div class="metric-grid" style="margin-bottom:18px">{metrics}</div>
  {hist_html}
</div>"""

    def _section_llm(self, name: str, cfg: dict, labels: pd.Series,
                     ground_truth_col: Optional[str]) -> str:
        counts = labels.value_counts()
        n = len(labels)
        n_cats = counts.shape[0]
        balance = round(counts.min() / counts.max(), 2) if counts.max() > 0 else 0
        entropy_val = self._entropy(counts.values / n)
        max_ent = np.log2(n_cats) if n_cats > 1 else 1
        norm_entropy = round(entropy_val / max_ent, 2) if max_ent else 0

        bar = _bar_chart(counts, f"Category distribution — {name}")
        donut = _donut_chart(counts, "Share by category")

        metrics = "".join([
            _metric(f"{n:,}", "Classified rows"),
            _metric(str(n_cats), "Categories"),
            _metric(f"{balance:.2f}", "Balance ratio",
                    "ok" if balance > 0.5 else "warn" if balance > 0.2 else "bad"),
            _metric(f"{norm_entropy:.2f}", "Norm. entropy",
                    "ok" if norm_entropy > 0.7 else "warn"),
        ])

        # Keywords
        text_col = self._best_text_col(cfg)
        kw_html = ""
        if text_col:
            kws = _top_keywords(self.df[text_col], labels)
            colors = _palette(len(kws))
            kw_html = f"""
<div class="card" style="margin-top:0">
  <h3>Top keywords per category (TF-IDF)</h3>
  {_keywords_html(kws, colors)}
</div>"""

        # Sample texts
        sample_html = ""
        if text_col:
            sample_html = f"""
<div class="card" style="margin-top:0">
  <h3>Sample texts per category</h3>
  {_sample_table(self.df.assign(**{f"_label": labels}), "_label", text_col)}
</div>"""

        # Ground truth
        gt_html = ""
        if ground_truth_col and ground_truth_col in self.df.columns:
            gt_html = self._section_ground_truth(labels, self.df[ground_truth_col].astype(str), name)

        return f"""
<div id="{name}">
  <div class="divider"></div>
  <h2>{name} &nbsp;<span style="font-weight:400;font-size:0.8em;color:#888">LLM classification</span></h2>
  <div class="metric-grid" style="margin-bottom:18px">{metrics}</div>
  <div class="chart-grid">
    <div class="chart-box">{bar}</div>
    <div class="chart-box">{donut}</div>
  </div>
  {kw_html}
  {sample_html}
  {gt_html}
</div>"""

    def _section_clustering(self, name: str, cfg: dict, labels: pd.Series) -> str:
        counts = labels.value_counts()
        n = len(labels)
        noise = int((labels == "-1").sum())

        bar = _bar_chart(counts, f"Cluster sizes — {name}")

        metrics = "".join([
            _metric(f"{n:,}", "Total rows"),
            _metric(str(counts.shape[0]), "Clusters"),
            _metric(f"{noise:,}", "Noise points (-1)"),
            _metric(f"{counts.max():,}", "Largest cluster"),
            _metric(f"{counts.min():,}", "Smallest cluster"),
        ])

        # 2D scatter
        text_col = self._best_text_col(cfg)
        scatter_html = ""
        if text_col and len(self.df) <= 50_000:
            try:
                x, y, l_sub = _project_2d(self.df[text_col], labels)
                scatter_html = f'<div class="chart-full">{_scatter_chart(x, y, l_sub.tolist(), f"2D projection — {name} (UMAP/PCA)")}</div>'
            except Exception as e:
                logger.warning(f"Could not build scatter for {name}: {e}")

        # Keywords
        kw_html = ""
        if text_col:
            kws = _top_keywords(self.df[text_col], labels)
            colors = _palette(len(kws))
            kw_html = f"""
<div class="card" style="margin-top:0">
  <h3>Top keywords per cluster (TF-IDF)</h3>
  {_keywords_html(kws, colors)}
</div>"""

        return f"""
<div id="{name}">
  <div class="divider"></div>
  <h2>{name} &nbsp;<span style="font-weight:400;font-size:0.8em;color:#888">clustering</span></h2>
  <div class="metric-grid" style="margin-bottom:18px">{metrics}</div>
  <div class="chart-full">{bar}</div>
  {scatter_html}
  {kw_html}
</div>"""

    def _section_cross(self, out_cols: Dict[str, str]) -> str:
        """Cross-perspective heatmap between all pairs of perspectives."""
        names = list(out_cols.keys())
        html_parts = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                col_a, col_b = out_cols[a], out_cols[b]
                ct = pd.crosstab(
                    self.df[col_a].fillna("Unknown").astype(str),
                    self.df[col_b].fillna("Unknown").astype(str),
                )
                chart = _heatmap_chart(ct, f"Cross-perspective: {a}  ×  {b}")
                html_parts.append(f'<div class="chart-full">{chart}</div>')

        return f"""
<div id="_cross">
  <div class="divider"></div>
  <h2>Cross-perspective analysis</h2>
  <p style="color:#666;font-size:0.88em;margin-bottom:16px">
    Each heatmap shows how rows classified by one perspective map to another —
    useful for spotting correlations and disagreements between methods.
  </p>
  {"".join(html_parts)}
</div>"""

    def _section_ground_truth(self, predicted: pd.Series, truth: pd.Series,
                               perspective_name: str) -> str:
        """Confusion matrix + classification report vs ground truth."""
        from sklearn.metrics import (
            classification_report, accuracy_score, confusion_matrix
        )

        # Align indices
        mask = truth.notna() & predicted.notna()
        y_true = truth[mask]
        y_pred = predicted[mask]

        acc = accuracy_score(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # Build metrics table
        rows_html = ""
        for label, row in report_dict.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            p = row.get("precision", 0)
            r = row.get("recall", 0)
            f = row.get("f1-score", 0)
            s = int(row.get("support", 0))
            f_cls = "ok" if f >= 0.8 else "warn" if f >= 0.5 else "bad"
            rows_html += f"""<tr>
              <td><b>{label}</b></td>
              <td>{p:.2f}</td><td>{r:.2f}</td>
              <td class="{f_cls}"><b>{f:.2f}</b></td>
              <td>{s:,}</td>
            </tr>"""

        wa = report_dict.get("weighted avg", {})
        summary_row = f"""<tr style="background:#f7f9ff;font-weight:600">
          <td>weighted avg</td>
          <td>{wa.get('precision',0):.2f}</td>
          <td>{wa.get('recall',0):.2f}</td>
          <td>{wa.get('f1-score',0):.2f}</td>
          <td>{int(wa.get('support',0)):,}</td>
        </tr>"""

        # Confusion matrix
        labels_order = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
        cm_html = _confusion_matrix_chart(cm_df)

        acc_cls = "ok" if acc >= 0.8 else "warn" if acc >= 0.5 else "bad"

        return f"""
<div class="card" style="margin-top:18px">
  <h3>Ground-truth evaluation &nbsp;<span style="font-size:0.85em;color:#888">vs <code>{perspective_name}</code></span></h3>
  <div class="metric-grid" style="margin-bottom:16px">
    {_metric(f"{acc:.1%}", "Accuracy", acc_cls)}
    {_metric(f"{wa.get('precision',0):.2f}", "Weighted precision")}
    {_metric(f"{wa.get('recall',0):.2f}", "Weighted recall")}
    {_metric(f"{wa.get('f1-score',0):.2f}", "Weighted F1", acc_cls)}
    {_metric(str(int(wa.get('support',0))), "Evaluated rows")}
  </div>
  <table>
    <thead><tr><th>Category</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
    <tbody>{rows_html}{summary_row}</tbody>
  </table>
  <div style="margin-top:18px">{cm_html}</div>
</div>"""

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _best_text_col(self, cfg: dict) -> Optional[str]:
        """Pick the first text column that exists in self.df."""
        cols = cfg.get("columns", self.text_columns)
        for c in cols:
            if c in self.df.columns:
                return c
            pc = f"{c}_preprocessed"
            if pc in self.df.columns:
                return pc
        return None

    @staticmethod
    def _entropy(probs: np.ndarray) -> float:
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))


# ── Convenience function ──────────────────────────────────────────────────────

def generate_report(
    df: pd.DataFrame,
    perspectives_config: dict,
    results_dir: str,
    text_columns: Optional[List[str]] = None,
    ground_truth_col: Optional[str] = None,
) -> str:
    """
    One-call interface.

    Returns the path to the saved HTML report.
    """
    reporter = HTMLReporter(df, perspectives_config, results_dir, text_columns)
    return reporter.generate(ground_truth_col=ground_truth_col)
