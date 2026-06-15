"""
╔══════════════════════════════════════════════════════════════╗
║   UHI Analysis Dashboard v3.0                                ║
║   Urban Heat Island Detection — Laem Chabang / GISTDA        ║
╠══════════════════════════════════════════════════════════════╣
║   Stack: Streamlit · GEE · Folium · rasterio · matplotlib   ║
║   วิธีรัน:                                                   ║
║   1. pip install -r requirements.txt                         ║
║   2. earthengine authenticate                                ║
║   3. streamlit run app.py                                    ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import io, base64, os, warnings, json
from datetime import date, timedelta
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# ESG PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_esg_pdf(
    avg_lst: float, max_lst: float, min_lst: float,
    uhi_int: float, std_lst: float,
    seg_stats: dict, zonal_rows: list,
    gee_info: str,
    lst_rgb: np.ndarray, seg_mask: np.ndarray, rgb_array: np.ndarray,
) -> bytes:
    """สร้าง ESG PDF Report จากผลวิเคราะห์ UHI"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import io as _io

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=16, spaceAfter=6, alignment=TA_CENTER)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
    h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
                                  fontSize=12, spaceBefore=14, spaceAfter=4,
                                  textColor=colors.HexColor("#1a3a5c"))
    body_style  = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=14)
    footer_style = ParagraphStyle("footer", parent=styles["Normal"],
                                   fontSize=7, textColor=colors.grey, alignment=TA_CENTER)

    tbl_style = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f5f8fb"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#c5d5e8")),
        ("ALIGN",          (1, 1), (-1, -1), "CENTER"),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ])

    story = []

    # ── Cover ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("ESG Environmental Report", title_style))
    story.append(Paragraph(
        "Urban Heat Island Analysis · Laem Chabang Industrial Estate", sub_style))
    story.append(Paragraph(
        f"Generated: {date.today().strftime('%d %B %Y')}  |  Data: {gee_info}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#1a3a5c"), spaceAfter=14))

    # ── 1. Key Metrics ─────────────────────────────────────────────────────────
    story.append(Paragraph("1. Key LST Metrics", h2_style))
    metrics_data = [
        ["Indicator",     "Value",          "Unit", "Note"],
        ["Mean LST",      f"{avg_lst:.2f}", "C",    "Median composite"],
        ["Max LST",       f"{max_lst:.2f}", "C",    "Peak heat"],
        ["Min LST",       f"{min_lst:.2f}", "C",    "Cool island"],
        ["UHI Intensity", f"{uhi_int:.2f}", "C",    "Max - Min"],
        ["LST Std Dev",   f"{std_lst:.2f}", "C",    "Spatial variability"],
    ]
    t1 = Table(metrics_data, colWidths=[4.5*cm, 2.5*cm, 2*cm, None])
    t1.setStyle(tbl_style)
    story.append(t1)
    story.append(Spacer(1, 0.4*cm))

    # ── 2. Land Cover ──────────────────────────────────────────────────────────
    story.append(Paragraph("2. Land Cover Classification", h2_style))
    lc_data = [
        ["Land Cover Class",           "Coverage (%)"],
        ["Built-up / Impervious",      f"{seg_stats['buildings']:.1f}%"],
        ["Vegetation / Green Area",    f"{seg_stats['vegetation']:.1f}%"],
        ["Water Body",                 f"{seg_stats['water']:.1f}%"],
    ]
    t2 = Table(lc_data, colWidths=[9*cm, 4*cm])
    t2.setStyle(tbl_style)
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    # ── 3. Zonal Statistics ────────────────────────────────────────────────────
    story.append(Paragraph("3. Zonal Mean LST by Land Cover", h2_style))
    z_header = ["Land Cover", "Pixels", "Mean LST", "Max LST", "Min LST", "Std Dev"]
    z_data   = [z_header]
    for z in zonal_rows:
        def _f(v): return f"{v:.2f} C" if v is not None else "-"
        z_data.append([
            z["class"], f"{z['pixels']:,}",
            _f(z["mean"]), _f(z["max"]), _f(z["min"]), _f(z["std"]),
        ])
    t3 = Table(z_data, colWidths=[3*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t3.setStyle(tbl_style)
    story.append(t3)
    story.append(Spacer(1, 0.4*cm))

    # ── 4. UHI Summary & ESG Remarks ──────────────────────────────────────────
    story.append(Paragraph("4. UHI Summary and ESG Remarks", h2_style))
    buildup_z = next((z for z in zonal_rows if z["class"] == "Built-up"),   None)
    veg_z     = next((z for z in zonal_rows if z["class"] == "Vegetation"), None)
    if buildup_z and veg_z and buildup_z["mean"] and veg_z["mean"]:
        delta = buildup_z["mean"] - veg_z["mean"]
        story.append(Paragraph(
            f"Built-up areas recorded a mean LST of <b>{buildup_z['mean']:.2f} C</b>, "
            f"compared to vegetation areas at <b>{veg_z['mean']:.2f} C</b>. "
            f"The Urban Heat Island differential is <b>{delta:+.2f} C</b>. "
            f"Peak UHI Intensity (Max-Min) = <b>{uhi_int:.2f} C</b>.",
            body_style,
        ))
        story.append(Spacer(1, 0.2*cm))

    esg_level = "HIGH" if uhi_int > 10 else ("MODERATE" if uhi_int > 5 else "LOW")
    esg_color = (
        "#c62828" if esg_level == "HIGH" else
        "#e65100" if esg_level == "MODERATE" else
        "#2e7d32"
    )
    story.append(Paragraph(
        f"ESG Heat Risk Level: <font color='{esg_color}'><b>{esg_level}</b></font>",
        body_style,
    ))
    story.append(Spacer(1, 0.25*cm))
    story.append(Paragraph(
        "Recommendations: Consider increasing green coverage (trees/vegetation) "
        "in high-LST zones, installing cool roofs (high-albedo materials), "
        "and incorporating water features to reduce surface heat accumulation.",
        body_style,
    ))
    story.append(Spacer(1, 0.4*cm))

    # ── 5. Visual Outputs ──────────────────────────────────────────────────────
    story.append(Paragraph("5. Visual Outputs", h2_style))

    def _np_to_rl_img(arr: np.ndarray, w_cm: float = 5.2) -> RLImage:
        tmp = _io.BytesIO()
        Image.fromarray(arr).save(tmp, format="PNG")
        tmp.seek(0)
        return RLImage(tmp, width=w_cm*cm, height=w_cm*0.75*cm)

    img_row = [_np_to_rl_img(rgb_array), _np_to_rl_img(seg_mask), _np_to_rl_img(lst_rgb)]
    img_tbl = Table([img_row], colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(img_tbl)

    cap_tbl = Table(
        [["Satellite RGB", "Land Classification", "LST Heatmap"]],
        colWidths=[5.5*cm, 5.5*cm, 5.5*cm],
    )
    cap_tbl.setStyle(TableStyle([
        ("FONTSIZE",  (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.grey),
        ("ALIGN",     (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(cap_tbl)

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        "This report was generated by the UHI Analysis System for ESG environmental "
        "disclosure purposes. LST data sourced from Landsat 8/9 Collection 2 Level-2 "
        "via Google Earth Engine.",
        footer_style,
    ))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UHI · Urban Heat Island Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIG (upload size)
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_config():
    cfg_dir  = os.path.join(os.path.dirname(__file__), ".streamlit")
    cfg_file = os.path.join(cfg_dir, "config.toml")
    os.makedirs(cfg_dir, exist_ok=True)
    if not os.path.exists(cfg_file):
        with open(cfg_file, "w") as f:
            f.write("[server]\nmaxUploadSize = 10240\n")

_ensure_config()

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
# Palette: deep navy base, thermal-orange heat accent, cyan sensor accent
# Signature: thermal scan-line effect on header + monospace telemetry numbers
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg0:     #050c18;
  --bg1:     #0a1628;
  --bg2:     #0f1e35;
  --bg3:     #152440;
  --border:  #1e3555;
  --heat:    #ff5722;
  --heat2:   #ff8f00;
  --cool:    #00bcd4;
  --veg:     #4caf50;
  --water:   #2196f3;
  --text:    #e8f4fd;
  --muted:   #6b8cad;
  --dim:     #3a5070;
}

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  color: var(--text);
}

.stApp { background: var(--bg0) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Header ── */
.hdr {
  position: relative;
  background: linear-gradient(160deg, #080f1f 0%, #0f1e35 60%, #0a1a30 100%);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 2rem 2.5rem 1.6rem;
  margin-bottom: 1.5rem;
  overflow: hidden;
}
.hdr::before {
  content: "";
  position: absolute; inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 3px,
    rgba(255,87,34,0.022) 3px,
    rgba(255,87,34,0.022) 4px
  );
  pointer-events: none;
}
.hdr::after {
  content: "";
  position: absolute;
  right: -60px; top: -60px;
  width: 280px; height: 280px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,87,34,0.12) 0%, transparent 70%);
  pointer-events: none;
}
.hdr-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem; letter-spacing: 0.18em;
  color: var(--heat2); text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.hdr h1 {
  font-size: 1.85rem; font-weight: 700;
  color: var(--text); margin: 0 0 0.4rem;
  letter-spacing: -0.03em; line-height: 1.15;
}
.hdr h1 span {
  background: linear-gradient(90deg, var(--heat), var(--heat2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hdr-sub { font-size: 0.82rem; color: var(--muted); margin: 0; }
.hdr-badges { display: flex; gap: 0.5rem; margin-top: 0.9rem; flex-wrap: wrap; }
.badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem; letter-spacing: 0.1em;
  padding: 3px 10px; border-radius: 4px;
  text-transform: uppercase;
}
.badge-gee   { background: rgba(0,188,212,0.12); border: 1px solid rgba(0,188,212,0.35); color: #00bcd4; }
.badge-l8    { background: rgba(76,175,80,0.1);  border: 1px solid rgba(76,175,80,0.3);  color: #66bb6a; }
.badge-ai    { background: rgba(255,87,34,0.1);  border: 1px solid rgba(255,87,34,0.3);  color: var(--heat); }
.badge-poc   { background: rgba(255,143,0,0.1);  border: 1px solid rgba(255,143,0,0.3);  color: var(--heat2); }

/* ── Section Label ── */
.sec-label {
  display: flex; align-items: center; gap: 0.5rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem; letter-spacing: 0.18em;
  color: var(--heat2); text-transform: uppercase;
  margin-bottom: 0.75rem;
}
.sec-label::after {
  content: "";
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Metric Grid ── */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem; }
.mc {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.1rem;
  position: relative; overflow: hidden;
}
.mc::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.mc-heat::before  { background: linear-gradient(90deg, var(--heat), var(--heat2)); }
.mc-cool::before  { background: linear-gradient(90deg, #0277bd, var(--cool)); }
.mc-veg::before   { background: linear-gradient(90deg, #2e7d32, var(--veg)); }
.mc-build::before { background: linear-gradient(90deg, #4a148c, #7b1fa2); }
.mc-uhi::before   { background: linear-gradient(90deg, var(--heat), #ff1744); }
.mc-scene::before { background: linear-gradient(90deg, #00796b, #00bfa5); }
.mc-lbl {
  font-size: 0.65rem; font-weight: 600;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 0.35rem;
}
.mc-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.55rem; font-weight: 600;
  color: var(--text); line-height: 1;
}
.mc-unit { font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; }

/* ── Status Pill ── */
.pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.7rem; font-weight: 500; font-family: 'JetBrains Mono', monospace;
  padding: 3px 11px; border-radius: 100px; margin-bottom: 1rem;
}
.pill-live { background: rgba(76,175,80,0.13); border: 1px solid rgba(76,175,80,0.35); color: #66bb6a; }
.pill-mock { background: rgba(0,188,212,0.12); border: 1px solid rgba(0,188,212,0.35); color: #00bcd4; }
.pill-warn { background: rgba(255,143,0,0.12); border: 1px solid rgba(255,143,0,0.35); color: var(--heat2); }
.dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: blink 2s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Cards ── */
.info-card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.1rem 1.3rem; margin-bottom: 0.75rem;
}
.info-card-title { font-size: 0.75rem; font-weight: 600; color: var(--cool); margin-bottom: 0.5rem; }

/* ── Layer Preview ── */
.layer-wrap { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 0.75rem; }
.layer-cap { font-size: 0.72rem; color: var(--muted); text-align: center; margin-top: 0.4rem; }

/* ── Legend ── */
.leg { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; }
.leg-i { display: flex; align-items: center; gap: 5px; font-size: 0.72rem; color: var(--muted); }
.leg-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }

/* ── Alert ── */
.alert-err  { background: rgba(244,67,54,0.08);  border: 1px solid rgba(244,67,54,0.3);  border-radius: 8px; padding: 0.85rem 1rem; color: #ef9a9a; font-size: 0.82rem; margin-bottom: 0.75rem; }
.alert-info { background: rgba(0,188,212,0.07);  border: 1px solid rgba(0,188,212,0.25); border-radius: 8px; padding: 0.85rem 1rem; color: #80deea; font-size: 0.82rem; margin-bottom: 0.75rem; }
.alert-ok   { background: rgba(76,175,80,0.08);  border: 1px solid rgba(76,175,80,0.3);  border-radius: 8px; padding: 0.85rem 1rem; color: #a5d6a7; font-size: 0.82rem; margin-bottom: 0.75rem; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--heat), var(--heat2)) !important;
  color: white !important; border: none !important;
  border-radius: 7px !important; font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  letter-spacing: 0.02em !important;
}
.stButton > button:hover { filter: brightness(1.1); }

/* ── Inputs ── */
div[data-testid="stFileUploader"] {
  background: var(--bg2) !important;
  border: 1px dashed var(--dim) !important;
  border-radius: 8px !important;
}
.stTextInput input, .stSelectbox select, .stDateInput input {
  background: var(--bg3) !important; border-color: var(--border) !important;
  color: var(--text) !important; border-radius: 6px !important;
}
.stSlider { accent-color: var(--heat); }
div[data-testid="stExpander"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
h2, h3, h4 { color: var(--text) !important; }
hr { border-color: var(--border) !important; }
label, .stSelectbox label, .stSlider label { color: var(--muted) !important; }

/* ── Sidebar sections ── */
.sb-sec {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem; letter-spacing: 0.2em;
  color: var(--heat2); text-transform: uppercase;
  margin: 1rem 0 0.5rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid var(--border);
}
.sb-card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 8px; padding: 0.85rem 1rem; margin-bottom: 0.75rem;
}
.sb-card-title { font-size: 0.72rem; font-weight: 600; color: var(--cool); margin-bottom: 0.5rem; }

/* ── ZonalStats Table ── */
.zt { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.zt th { background: var(--bg3); color: var(--muted); font-weight: 500; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid var(--border); }
.zt td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
.zt tr:last-child td { border-bottom: none; }
.zt tr:hover td { background: var(--bg3); }
.zt .hot { color: var(--heat); }
.zt .ok  { color: var(--veg); }
.zt .mid { color: var(--heat2); }

/* ── GEE pipeline diagram ── */
.pipe { display: flex; align-items: center; gap: 0; flex-wrap: wrap; margin: 0.5rem 0; }
.pipe-box { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 0.3rem 0.65rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--cool); white-space: nowrap; }
.pipe-arr { color: var(--dim); font-size: 1rem; padding: 0 0.3rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-eyebrow">🛰️ GISTDA · Cal/Val · POC v3.0</div>
  <h1>Urban <span>Heat Island</span> Analysis</h1>
  <p class="hdr-sub">อัปโหลด GeoTIFF (THEOS-2 / Sentinel / True-color) → AI Land Classification → LST จาก Landsat 8/9 via GEE → Interactive Layer Map</p>
  <div class="hdr-badges">
    <span class="badge badge-gee">Google Earth Engine</span>
    <span class="badge badge-l8">Landsat 8/9 C2 L2</span>
    <span class="badge badge-ai">Rule-based Seg</span>
    <span class="badge badge-poc">Laem Chabang · Chonburi</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# GEE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _gee_init(project: str):
    """
    Initialize GEE — cached per project string.
    ต้อง earthengine authenticate ก่อนใช้ครั้งแรก
    """
    try:
        import ee
        if project.strip():
            ee.Initialize(project=project.strip())
        else:
            ee.Initialize()
        return True, "OK"
    except Exception as e:
        return False, str(e)


def fetch_lst_gee(
    bounds: dict,
    date_start: str,
    date_end: str,
    cloud_pct: int = 20,
    scale: int = 30,
    lst_cmap: str = "inferno",
) -> tuple[np.ndarray, str, str | None]:
    """
    ดึง LST (°C) จาก Landsat 8/9 C2 L2 ผ่าน GEE

    Returns TWO outputs:
      1. numpy array (float32 °C) — สำหรับ stats / zonal analysis
      2. GEE Tile URL (str | None) — สำหรับ Folium TileLayer (smooth, native res)

    Pipeline:
      ST_B10 DN → ×0.00341802 +149.0 → K → −273.15 → °C
      QA_PIXEL bit3/4 cloud+shadow mask → median composite → clip ROI
      → getDownloadURL (numpy) + getMapId (tile URL)
    """
    import ee, requests, rasterio
    from rasterio.io import MemoryFile

    # Colormap palette สำหรับ GEE visualize (16 stops จาก matplotlib cmap)
    _CMAP_PALETTES = {
        "inferno":   ["000004", "1b0c41", "4a0c6b", "781c6d", "a52c60",
                      "cf4446", "ed6925", "f98e09", "fdc328", "fcffa4"],
        "hot":       ["000000", "330000", "660000", "990000", "cc0000",
                      "ff0000", "ff3300", "ff6600", "ff9900", "ffcc00", "ffffff"],
        "plasma":    ["0d0887", "41049d", "6a00a8", "8f0da4", "b12a90",
                      "cc4778", "e16462", "f2844b", "fca636", "fcce25", "f0f921"],
        "RdYlBu_r":  ["313695", "4575b4", "74add1", "abd9e9", "e0f3f8",
                      "ffffbf", "fee090", "fdae61", "f46d43", "d73027", "a50026"],
        "coolwarm":  ["3b4cc0", "6788ee", "9bbcff", "c9d8ef", "edddd4",
                      "f7b89c", "e8755a", "c84b31", "b40426"],
        "magma":     ["000004", "180f3d", "440f76", "721f81", "9e2f7f",
                      "cd4071", "f1605d", "fd9668", "feca8d", "fcfdbf"],
    }
    palette = _CMAP_PALETTES.get(lst_cmap, _CMAP_PALETTES["inferno"])

    roi = ee.Geometry.Rectangle(
        [bounds["west"], bounds["south"], bounds["east"], bounds["north"]]
    )

    def _mask_clouds(img):
        qa     = img.select("QA_PIXEL")
        cloud  = qa.bitwiseAnd(1 << 3).eq(0)
        shadow = qa.bitwiseAnd(1 << 4).eq(0)
        return img.updateMask(cloud.And(shadow))

    def _to_celsius(img):
        lst_c = (img.select("ST_B10")
                    .multiply(0.00341802)
                    .add(149.0)
                    .subtract(273.15)
                    .rename("LST_C"))
        return lst_c.copyProperties(img, img.propertyNames())

    col8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterBounds(roi)
              .filterDate(date_start, date_end)
              .filter(ee.Filter.lt("CLOUD_COVER", cloud_pct))
              .select(["ST_B10", "QA_PIXEL"]))

    col9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
              .filterBounds(roi)
              .filterDate(date_start, date_end)
              .filter(ee.Filter.lt("CLOUD_COVER", cloud_pct))
              .select(["ST_B10", "QA_PIXEL"]))

    merged = col8.merge(col9)
    n_scenes = merged.size().getInfo()

    if n_scenes == 0:
        raise ValueError(
            f"ไม่พบ Landsat scene ใน ROI ช่วง {date_start} → {date_end} "
            f"(cloud ≤ {cloud_pct}%)\n"
            "ลอง: ขยาย date range / เพิ่ม cloud_pct / ตรวจสอบ bounds"
        )

    lst_median = (merged
                  .map(_mask_clouds)
                  .map(_to_celsius)
                  .select("LST_C")
                  .median()
                  .clip(roi))

    # ── 1. Download numpy array for stats ──────────────────────────────────
    download_url = lst_median.getDownloadURL({
        "region": roi,
        "scale":  scale,
        "format": "GEO_TIFF",
        "crs":    "EPSG:4326",
    })
    resp = requests.get(download_url, timeout=300)
    resp.raise_for_status()

    with MemoryFile(resp.content) as mf:
        with mf.open() as ds:
            arr = ds.read(1).astype(np.float32)

    # ── 2. Get smooth Tile URL from GEE (เหมือน Code Editor) ───────────────
    # คำนวณ min/max จาก array ที่ download มาแล้ว เพื่อ stretch ที่ถูกต้อง
    valid    = arr[np.isfinite(arr)]
    lst_min  = float(np.percentile(valid, 2))
    lst_max  = float(np.percentile(valid, 98))

    tile_url = None
    try:
        map_id = lst_median.visualize(
            min=lst_min,
            max=lst_max,
            palette=palette,
        ).getMapId()
        # GEE tile URL template
        tile_url = map_id["tile_fetcher"].url_format
    except Exception:
        # Fallback: ถ้า getMapId ไม่ทำงาน (credential scope) ใช้ numpy overlay แทน
        tile_url = None

    info = (f"{n_scenes} scenes · {date_start} → {date_end} · "
            f"cloud ≤ {cloud_pct}% · res {scale}m · "
            f"LST range {lst_min:.1f}–{lst_max:.1f}°C")
    return arr, info, tile_url


def fetch_ndvi_gee(bounds: dict, date_start: str, date_end: str,
                   cloud_pct: int = 20, scale: int = 30) -> np.ndarray | None:
    """
    ดึง NDVI จาก Landsat 8/9 (Band 5 = NIR, Band 4 = Red) ผ่าน GEE
    เป็น optional layer สำหรับ overlay
    """
    try:
        import ee, requests, rasterio
        from rasterio.io import MemoryFile

        roi = ee.Geometry.Rectangle(
            [bounds["west"], bounds["south"], bounds["east"], bounds["north"]]
        )

        def _mask_clouds(img):
            qa = img.select("QA_PIXEL")
            return img.updateMask(
                qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
            )

        def _ndvi(img):
            nir  = img.select("SR_B5").multiply(0.0000275).add(-0.2)
            red  = img.select("SR_B4").multiply(0.0000275).add(-0.2)
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
            return ndvi.copyProperties(img, img.propertyNames())

        col8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                  .filterBounds(roi)
                  .filterDate(date_start, date_end)
                  .filter(ee.Filter.lt("CLOUD_COVER", cloud_pct))
                  .select(["SR_B4", "SR_B5", "QA_PIXEL"]))
        col9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                  .filterBounds(roi)
                  .filterDate(date_start, date_end)
                  .filter(ee.Filter.lt("CLOUD_COVER", cloud_pct))
                  .select(["SR_B4", "SR_B5", "QA_PIXEL"]))

        merged = col8.merge(col9)
        ndvi_img = (merged.map(_mask_clouds).map(_ndvi)
                    .select("NDVI").median().clip(roi))

        url = ndvi_img.getDownloadURL({
            "region": roi, "scale": scale,
            "format": "GEO_TIFF", "crs": "EPSG:4326",
        })
        resp = __import__("requests").get(url, timeout=300)
        resp.raise_for_status()

        with MemoryFile(resp.content) as mf:
            with mf.open() as ds:
                return ds.read(1).astype(np.float32)
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

MAX_PX = 2048  # display downsample cap


def load_geotiff(file_bytes: bytes) -> tuple[np.ndarray, dict]:
    """
    โหลด GeoTIFF → (rgb uint8 H×W×3, bounds WGS-84)
    รองรับ 1-band (panchromatic), 3-band, 4-band (drop alpha)
    Downsample ถ้าใหญ่กว่า MAX_PX
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.enums import Resampling
    from pyproj import Transformer

    with MemoryFile(file_bytes) as mf:
        with mf.open() as ds:
            H, W = ds.height, ds.width
            crs  = ds.crs

            # Convert bounds to WGS-84
            if crs and not crs.is_geographic:
                tfm = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                west, south = tfm.transform(ds.bounds.left,  ds.bounds.bottom)
                east, north = tfm.transform(ds.bounds.right, ds.bounds.top)
            else:
                west, south = ds.bounds.left,  ds.bounds.bottom
                east, north = ds.bounds.right, ds.bounds.top

            bounds = dict(west=west, south=south, east=east, north=north)

            # Downsample
            scale = min(1.0, MAX_PX / max(H, W))
            oh, ow = max(1, int(H * scale)), max(1, int(W * scale))

            n = ds.count
            if n >= 3:
                raw = ds.read([1, 2, 3], out_shape=(3, oh, ow),
                              resampling=Resampling.bilinear)
                rgb = np.stack([raw[0], raw[1], raw[2]], axis=-1)
            else:
                raw = ds.read(1, out_shape=(oh, ow),
                              resampling=Resampling.bilinear)
                rgb = np.stack([raw] * 3, axis=-1)

    # 2-percentile stretch → uint8
    rgb = rgb.astype(float)
    for c in range(3):
        mn, mx = np.percentile(rgb[:, :, c], 2), np.percentile(rgb[:, :, c], 98)
        if mx > mn:
            rgb[:, :, c] = np.clip((rgb[:, :, c] - mn) / (mx - mn) * 255, 0, 255)
        else:
            rgb[:, :, c] = 0

    return rgb.astype(np.uint8), bounds


def segment_image(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Rule-based land classification:
      Class 0 — Built-up / Impervious (grey, high brightness)
      Class 1 — Vegetation (green-dominant)
      Class 2 — Water (blue-dominant)
    Returns: (colored_mask uint8 H×W×3, stats dict)
    """
    H, W = rgb.shape[:2]
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    veg_mask   = (g > r + 18) & (g > b + 18) & (g > 60)
    water_mask = (b > r + 18) & (b > g + 8)  & (b > 55)

    seg = np.zeros((H, W), dtype=np.uint8)   # 0 = built-up (default)
    seg[veg_mask]   = 1
    seg[water_mask] = 2

    COLORS = {
        0: [220, 60,  60 ],   # built-up → red
        1: [50,  180, 80 ],   # vegetation → green
        2: [40,  130, 220],   # water → blue
    }
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, col in COLORS.items():
        colored[seg == cls] = col

    total = H * W
    stats = {
        "buildings" : float((seg == 0).sum() / total * 100),
        "vegetation": float((seg == 1).sum() / total * 100),
        "water"     : float((seg == 2).sum() / total * 100),
    }
    return colored, stats, seg


def colorize_lst(arr: np.ndarray, cmap_name: str = "inferno") -> np.ndarray:
    """
    float32 LST array (°C) → uint8 RGBA heatmap
    NaN/nodata pixels → alpha=0 (fully transparent) ไม่ใช่สีดำ
    Returns H×W×4 RGBA
    """
    valid = arr[np.isfinite(arr)]
    if not len(valid):
        return np.zeros((*arr.shape, 4), dtype=np.uint8)
    mn, mx = np.percentile(valid, 2), np.percentile(valid, 98)
    norm   = np.clip((arr - mn) / (mx - mn + 1e-9), 0, 1)
    valid_mask = np.isfinite(arr)
    norm   = np.where(valid_mask, norm, 0.0)
    cmap   = matplotlib.colormaps[cmap_name]
    rgba   = (cmap(norm) * 255).astype(np.uint8)          # H×W×4
    rgba[:, :, 3] = np.where(valid_mask, 255, 0).astype(np.uint8)  # NaN → transparent
    return rgba  # RGBA


def colorize_ndvi(arr: np.ndarray) -> np.ndarray:
    """float32 NDVI [-1,1] → uint8 RGB (RdYlGn)"""
    norm = np.clip((arr + 1) / 2, 0, 1)
    norm = np.where(np.isfinite(arr), norm, 0)
    cmap = matplotlib.colormaps["RdYlGn"]
    return (cmap(norm)[:, :, :3] * 255).astype(np.uint8)


def arr_to_png_b64(arr: np.ndarray, alpha: int = 210) -> str:
    """
    numpy uint8 (H×W×3 RGB หรือ H×W×4 RGBA) → base64 PNG data-URI

    - RGB input  → apply uniform alpha (legacy behavior)
    - RGBA input → multiply existing alpha ด้วย (alpha/255)
      ทำให้ NaN pixels ที่มี alpha=0 ยังคง transparent แม้ scale ลง
    """
    if arr.ndim == 3 and arr.shape[2] == 4:
        # RGBA: scale alpha channel by overlay_alpha factor
        img = Image.fromarray(arr, mode="RGBA")
        r, g, b, a_ch = img.split()
        # Scale: a_final = a_original × (alpha / 255)
        a_arr = (np.array(a_ch).astype(float) * alpha / 255).clip(0, 255).astype(np.uint8)
        a_ch  = Image.fromarray(a_arr)
        merged = Image.merge("RGBA", (r, g, b, a_ch))
    else:
        # RGB: uniform alpha
        img  = Image.fromarray(arr).convert("RGBA")
        r, g, b, _ = img.split()
        a    = Image.fromarray(np.full(arr.shape[:2], alpha, dtype=np.uint8))
        merged = Image.merge("RGBA", (r, g, b, a))
    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def resize_to_match(src: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Resize float32 LST array (H×W) to match target shape.
    ใช้ rasterio reproject แทน PIL เพื่อ:
      - preserve NaN/nodata correctly (PIL converts NaN → 0)
      - bilinear interpolation บน float32 โดยตรง
      - ไม่มี blocky artifact จาก uint8 intermediate
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling as RioResampling
    from rasterio.io import MemoryFile

    th, tw = target.shape[:2]
    sh, sw = src.shape[:2]

    if (sh, sw) == (th, tw):
        return src.copy()

    # สร้าง dummy transform (geographic extent ไม่สำคัญ — แค่ resize pixel grid)
    src_transform = from_bounds(0, 0, 1, 1, sw, sh)
    dst_transform = from_bounds(0, 0, 1, 1, tw, th)

    dst = np.full((th, tw), np.nan, dtype=np.float32)

    reproject(
        source      = src,
        destination = dst,
        src_transform  = src_transform,
        dst_transform  = dst_transform,
        src_crs     = "EPSG:4326",
        dst_crs     = "EPSG:4326",
        src_nodata  = np.nan,
        dst_nodata  = np.nan,
        resampling  = RioResampling.bilinear,
    )
    return dst


def compute_zonal_stats(lst: np.ndarray, seg: np.ndarray) -> list[dict]:
    """
    Zonal statistics: mean/max/min LST per land-cover class
    """
    class_names = {0: "Built-up", 1: "Vegetation", 2: "Water"}
    rows = []
    for cls_id, cls_name in class_names.items():
        mask  = (seg == cls_id) & np.isfinite(lst)
        vals  = lst[mask]
        if len(vals) == 0:
            rows.append({"class": cls_name, "pixels": 0,
                         "mean": None, "max": None, "min": None, "std": None})
        else:
            rows.append({
                "class"  : cls_name,
                "pixels" : int(len(vals)),
                "mean"   : float(np.mean(vals)),
                "max"    : float(np.max(vals)),
                "min"    : float(np.min(vals)),
                "std"    : float(np.std(vals)),
            })
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════

_defaults = dict(
    processed    = False,
    rgb_array    = None,
    seg_mask     = None,
    seg_raw      = None,       # uint8 class index array
    lst_raw      = None,
    lst_rgb      = None,       # RGBA numpy array (NaN → alpha=0)
    lst_tile_url = None,       # GEE smooth tile URL (preferred for map)
    ndvi_raw     = None,
    ndvi_rgb     = None,
    seg_stats    = {},
    zonal_rows   = [],
    bounds       = None,
    center       = None,
    lst_source   = "",
    gee_info     = "",
    error_msg    = "",
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sb-sec">📡 Input Data</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Satellite GeoTIFF",
        type=["tif", "tiff"],
        help="RGB composite จาก THEOS-2, Sentinel-2, หรือ Landsat · รองรับถึง 10 GB",
    )
    st.caption("✅ รองรับ: 1-band / 3-band / 4-band · สูงสุด 10 GB")

    st.markdown('<div class="sb-sec">🌍 GEE · Landsat LST</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-card">', unsafe_allow_html=True)
    st.markdown('<div class="sb-card-title">🛰️ Google Earth Engine Config</div>', unsafe_allow_html=True)

    gee_project = st.text_input(
        "Project ID",
        value="",
        placeholder="your-gee-project-id",
        help="ใส่ GEE Cloud Project ที่ enabled Earth Engine API แล้ว",
    )

    d_default_end   = date.today()
    d_default_start = d_default_end - timedelta(days=90)

    c1, c2 = st.columns(2)
    with c1:
        date_start = st.date_input("เริ่มต้น", value=d_default_start)
    with c2:
        date_end   = st.date_input("สิ้นสุด",  value=d_default_end)

    cloud_pct = st.slider("Cloud Cover สูงสุด (%)", 0, 50, 20, 5)
    gee_scale = st.slider("Export Resolution (m)", 10, 120, 30, 10,
                          help="30m = Landsat native · เพิ่มถ้า area ใหญ่มาก")

    fetch_ndvi = st.checkbox("ดึง NDVI ด้วย (ใช้เวลาเพิ่ม)", value=False)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">🎨 Visualization</div>', unsafe_allow_html=True)

    lst_cmap = st.selectbox(
        "LST Colormap",
        ["inferno", "hot", "plasma", "RdYlBu_r", "coolwarm", "magma"],
    )
    overlay_alpha = st.slider("Overlay Opacity", 50, 255, 195, 5)
    basemap = st.selectbox(
        "Basemap",
        ["CartoDB dark_matter", "OpenStreetMap", "CartoDB positron",
         "Stamen Terrain"],
    )

    st.markdown('<div class="sb-sec">ℹ️ Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card">
      <div class="sb-card-title">GEE LST Pipeline</div>
      <div class="pipe">
        <div class="pipe-box">ST_B10 DN</div>
        <div class="pipe-arr">→</div>
        <div class="pipe-box">×0.00341802 +149</div>
        <div class="pipe-arr">→</div>
        <div class="pipe-box">Kelvin</div>
        <div class="pipe-arr">→</div>
        <div class="pipe-box">−273.15 → °C</div>
      </div>
      <div style="font-size:.68rem;color:#3a5070;margin-top:.5rem">
        QA_PIXEL bit3(cloud) bit4(shadow) mask → median composite
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# RUN BUTTON
# ═════════════════════════════════════════════════════════════════════════════

run_col, _ = st.columns([1, 4])
with run_col:
    run_btn = st.button("▶ Run Analysis", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

if run_btn:
    # ── Validation ───────────────────────────────────────────────────────────
    errors = []
    if uploaded_file is None:
        errors.append("กรุณาอัปโหลด GeoTIFF ก่อนรัน")
    if not gee_project.strip():
        errors.append("กรุณาใส่ GEE Project ID")
    if date_start >= date_end:
        errors.append("วันเริ่มต้นต้องก่อนวันสิ้นสุด")

    if errors:
        for e in errors:
            st.markdown(f'<div class="alert-err">⚠️ {e}</div>', unsafe_allow_html=True)
    else:
        st.session_state.error_msg = ""
        progress = st.progress(0, text="⏳ กำลังโหลด GeoTIFF...")

        # ── Step 1: Load GeoTIFF ─────────────────────────────────────────────
        try:
            with st.spinner("📂 อ่าน GeoTIFF..."):
                raw_bytes          = uploaded_file.read()
                rgb_arr, bounds    = load_geotiff(raw_bytes)
                center = [
                    (bounds["south"] + bounds["north"]) / 2,
                    (bounds["west"]  + bounds["east"])  / 2,
                ]
            progress.progress(20, text="✅ โหลดภาพสำเร็จ · กำลัง Segment...")
        except Exception as e:
            st.markdown(f'<div class="alert-err">❌ โหลด GeoTIFF ล้มเหลว: {e}</div>',
                        unsafe_allow_html=True)
            st.stop()

        # ── Step 2: Segmentation ─────────────────────────────────────────────
        try:
            seg_mask, seg_stats, seg_raw = segment_image(rgb_arr)
            progress.progress(40, text="✅ Segmentation เสร็จ · กำลังดึง LST จาก GEE...")
        except Exception as e:
            st.markdown(f'<div class="alert-err">❌ Segmentation ล้มเหลว: {e}</div>',
                        unsafe_allow_html=True)
            st.stop()

        # ── Step 3: GEE Init ─────────────────────────────────────────────────
        gee_ok, gee_msg = _gee_init(gee_project.strip())
        if not gee_ok:
            st.markdown(
                f'<div class="alert-err">❌ GEE Init ล้มเหลว: {gee_msg}<br>'
                f'ตรวจสอบ: (1) earthengine authenticate ครบ (2) Project ID ถูกต้อง (3) Earth Engine API enabled</div>',
                unsafe_allow_html=True)
            st.stop()

        progress.progress(50, text="🌍 GEE เชื่อมต่อสำเร็จ · กำลังดึง Landsat LST...")

        # ── Step 4: Fetch LST ────────────────────────────────────────────────
        try:
            lst_raw, gee_info = fetch_lst_gee(
                bounds,
                str(date_start),
                str(date_end),
                cloud_pct=cloud_pct,
                scale=gee_scale,
            )
            lst_source = f"Landsat 8/9 via GEE · {gee_info}"
            progress.progress(75, text="✅ ได้ LST แล้ว · กำลัง resize + colorize...")
        except Exception as e:
            st.markdown(f'<div class="alert-err">❌ ดึง LST จาก GEE ล้มเหลว: {e}</div>',
                        unsafe_allow_html=True)
            st.stop()

        # ── Step 5: Resize LST to match RGB ─────────────────────────────────
        if lst_raw.shape != rgb_arr.shape[:2]:
            lst_raw_disp = resize_to_match(lst_raw, rgb_arr)
        else:
            lst_raw_disp = lst_raw

        lst_rgb_colored = colorize_lst(lst_raw_disp, lst_cmap)

        # ── Step 6: NDVI (optional) ──────────────────────────────────────────
        ndvi_raw_disp = None
        ndvi_rgb_colored = None
        if fetch_ndvi:
            progress.progress(82, text="🌿 กำลังดึง NDVI จาก GEE...")
            ndvi_raw = fetch_ndvi_gee(
                bounds, str(date_start), str(date_end),
                cloud_pct=cloud_pct, scale=gee_scale,
            )
            if ndvi_raw is not None:
                if ndvi_raw.shape != rgb_arr.shape[:2]:
                    ndvi_raw_disp = resize_to_match(ndvi_raw, rgb_arr)
                else:
                    ndvi_raw_disp = ndvi_raw
                ndvi_rgb_colored = colorize_ndvi(ndvi_raw_disp)

        # ── Step 7: Zonal Statistics ─────────────────────────────────────────
        zonal_rows = compute_zonal_stats(lst_raw_disp, seg_raw)

        # ── Commit to session state ──────────────────────────────────────────
        st.session_state.update(dict(
            processed        = True,
            rgb_array        = rgb_arr,
            seg_mask         = seg_mask,
            seg_raw          = seg_raw,
            lst_raw          = lst_raw_disp,
            lst_rgb          = lst_rgb_colored,
            ndvi_raw         = ndvi_raw_disp,
            ndvi_rgb         = ndvi_rgb_colored,
            seg_stats        = seg_stats,
            zonal_rows       = zonal_rows,
            bounds           = bounds,
            center           = center,
            lst_source       = lst_source,
            gee_info         = gee_info,
        ))

        progress.progress(100, text="✅ เสร็จสิ้น!")
        st.markdown('<div class="alert-ok">✅ วิเคราะห์สำเร็จ — เลื่อนลงดูผลลัพธ์</div>',
                    unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# IDLE STATE (ก่อน run)
# ═════════════════════════════════════════════════════════════════════════════

if not st.session_state.processed:
    st.markdown("""
    <div style="background:#0a1628;border:1px solid #1e3555;border-radius:14px;
                padding:3rem;text-align:center;color:#3a5070;margin-top:1rem;">
      <div style="font-size:3rem;margin-bottom:1rem">🛰️</div>
      <div style="font-size:1rem;font-weight:600;color:#6b8cad;margin-bottom:.5rem">
        รอข้อมูล
      </div>
      <div style="font-size:.82rem">
        1. อัปโหลด GeoTIFF ใน sidebar<br>
        2. ใส่ GEE Project ID + ช่วงวันที่<br>
        3. กด <strong style="color:#ff8f00">▶ Run Analysis</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═════════════════════════════════════════════════════════════════════════════

ss = st.session_state
stats     = ss.seg_stats
lst_raw   = ss.lst_raw
bounds    = ss.bounds
center    = ss.center

avg_lst  = float(np.nanmean(lst_raw))
max_lst  = float(np.nanmax(lst_raw))
min_lst  = float(np.nanmin(lst_raw))
std_lst  = float(np.nanstd(lst_raw))
uhi_int  = max_lst - min_lst


# ── Status ────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="pill pill-live"><span class="dot"></span>'
    f'GEE · LANDSAT LIVE&nbsp;&nbsp;|&nbsp;&nbsp;{ss.gee_info}</div>',
    unsafe_allow_html=True
)


# ── Metric Cards ──────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">📊 Key Metrics</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="metric-grid">
  <div class="mc mc-heat">
    <div class="mc-lbl">Mean LST</div>
    <div class="mc-val">{avg_lst:.1f}°</div>
    <div class="mc-unit">Celsius · Landsat B10</div>
  </div>
  <div class="mc mc-heat">
    <div class="mc-lbl">Max LST</div>
    <div class="mc-val">{max_lst:.1f}°</div>
    <div class="mc-unit">Peak Urban Heat</div>
  </div>
  <div class="mc mc-cool">
    <div class="mc-lbl">Min LST</div>
    <div class="mc-val">{min_lst:.1f}°</div>
    <div class="mc-unit">Cool Island</div>
  </div>
  <div class="mc mc-uhi">
    <div class="mc-lbl">UHI Intensity</div>
    <div class="mc-val">{uhi_int:.1f}°</div>
    <div class="mc-unit">ΔT (Max − Min)</div>
  </div>
  <div class="mc mc-build">
    <div class="mc-lbl">Built-up</div>
    <div class="mc-val">{stats['buildings']:.1f}<span style="font-size:1rem">%</span></div>
    <div class="mc-unit">Impervious Surface</div>
  </div>
  <div class="mc mc-veg">
    <div class="mc-lbl">Vegetation</div>
    <div class="mc-val">{stats['vegetation']:.1f}<span style="font-size:1rem">%</span></div>
    <div class="mc-unit">Green Coverage</div>
  </div>
  <div class="mc mc-cool">
    <div class="mc-lbl">Water</div>
    <div class="mc-val">{stats['water']:.1f}<span style="font-size:1rem">%</span></div>
    <div class="mc-unit">Open Water</div>
  </div>
  <div class="mc mc-scene">
    <div class="mc-lbl">LST Std Dev</div>
    <div class="mc-val">{std_lst:.2f}°</div>
    <div class="mc-unit">Spatial Variability</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Layer Preview ─────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">🔬 Layer Preview</div>', unsafe_allow_html=True)

n_layers = 4 if (ss.ndvi_rgb is not None) else 3
cols_prev = st.columns(n_layers)

with cols_prev[0]:
    st.markdown('<div class="layer-wrap">', unsafe_allow_html=True)
    st.image(ss.rgb_array, use_column_width=True)
    st.markdown('<div class="layer-cap">📡 Satellite Image (RGB)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with cols_prev[1]:
    st.markdown('<div class="layer-wrap">', unsafe_allow_html=True)
    st.image(ss.seg_mask, use_column_width=True)
    st.markdown("""
    <div class="leg">
      <div class="leg-i"><div class="leg-dot" style="background:#dc3c3c"></div>Built-up</div>
      <div class="leg-i"><div class="leg-dot" style="background:#32b450"></div>Vegetation</div>
      <div class="leg-i"><div class="leg-dot" style="background:#2882dc"></div>Water</div>
    </div>
    <div class="layer-cap">🤖 Land Classification</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with cols_prev[2]:
    st.markdown('<div class="layer-wrap">', unsafe_allow_html=True)
    st.image(ss.lst_rgb, use_column_width=True)
    st.markdown(f"""
    <div class="leg">
      <div class="leg-i"><div class="leg-dot" style="background:#1a0033"></div>Cool</div>
      <div class="leg-i"><div class="leg-dot" style="background:#ff5500"></div>Warm</div>
      <div class="leg-i"><div class="leg-dot" style="background:#ffff80"></div>Hot</div>
    </div>
    <div class="layer-cap">🌡️ LST Heatmap ({lst_cmap})</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if ss.ndvi_rgb is not None:
    with cols_prev[3]:
        st.markdown('<div class="layer-wrap">', unsafe_allow_html=True)
        st.image(ss.ndvi_rgb, use_column_width=True)
        st.markdown("""
        <div class="leg">
          <div class="leg-i"><div class="leg-dot" style="background:#d73027"></div>Low</div>
          <div class="leg-i"><div class="leg-dot" style="background:#fee08b"></div>Mid</div>
          <div class="leg-i"><div class="leg-dot" style="background:#1a9850"></div>High</div>
        </div>
        <div class="layer-cap">🌿 NDVI (Landsat)</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


st.divider()


# ── Interactive Folium Map ─────────────────────────────────────────────────────
st.markdown('<div class="sec-label">🗺️ Interactive Map · Layer Control</div>',
            unsafe_allow_html=True)
st.caption("ใช้ Layer Control มุมบนขวาเพื่อ toggle แต่ละ layer")

import folium
from streamlit_folium import st_folium

folium_bounds = [
    [bounds["south"], bounds["west"]],
    [bounds["north"], bounds["east"]],
]

m = folium.Map(
    location=center,
    zoom_start=14,
    tiles=basemap,
)

# ── Layer 1: Satellite RGB ──
fg_rgb = folium.FeatureGroup(name="📡 Satellite Image", show=True)
folium.raster_layers.ImageOverlay(
    image=arr_to_png_b64(ss.rgb_array, 255),
    bounds=folium_bounds,
    opacity=0.9,
    zindex=1,
    cross_origin=False,
).add_to(fg_rgb)
fg_rgb.add_to(m)

# ── Layer 2: Segmentation Mask ──
fg_seg = folium.FeatureGroup(name="🤖 Land Classification", show=False)
folium.raster_layers.ImageOverlay(
    image=arr_to_png_b64(ss.seg_mask, overlay_alpha),
    bounds=folium_bounds,
    opacity=1.0,
    zindex=2,
    cross_origin=False,
).add_to(fg_seg)
fg_seg.add_to(m)

# ── Layer 3: LST Heatmap ──
fg_lst = folium.FeatureGroup(name="🌡️ LST Heatmap (Landsat B10)", show=False)
folium.raster_layers.ImageOverlay(
    image=arr_to_png_b64(ss.lst_rgb, overlay_alpha),
    bounds=folium_bounds,
    opacity=1.0,
    zindex=3,
    cross_origin=False,
).add_to(fg_lst)
fg_lst.add_to(m)

# ── Layer 4: NDVI (optional) ──
if ss.ndvi_rgb is not None:
    fg_ndvi = folium.FeatureGroup(name="🌿 NDVI (Landsat)", show=False)
    folium.raster_layers.ImageOverlay(
        image=arr_to_png_b64(ss.ndvi_rgb, overlay_alpha),
        bounds=folium_bounds,
        opacity=1.0,
        zindex=4,
        cross_origin=False,
    ).add_to(fg_ndvi)
    fg_ndvi.add_to(m)

# ── LayerControl — MUST be added last ──
folium.LayerControl(position="topright", collapsed=False).add_to(m)

# ── Measure Control ──
try:
    folium.plugins.MeasureControl(
        position="bottomleft",
        primary_length_unit="meters",
        secondary_length_unit="kilometers",
        primary_area_unit="sqmeters",
    ).add_to(m)
except Exception:
    pass

# ── Minimap ──
try:
    from folium.plugins import MiniMap
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
except Exception:
    pass

m.fit_bounds(folium_bounds)
st_folium(m, width="100%", height=560, returned_objects=[])


st.divider()


# ── Charts Row ────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-label">📈 Analysis Charts</div>', unsafe_allow_html=True)

fig_bg = "#0a1628"
ax_bg  = "#0f1e35"
tc     = "#e8f4fd"
mc     = "#6b8cad"

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), facecolor=fig_bg)
fig.patch.set_facecolor(fig_bg)

for ax in axes:
    ax.set_facecolor(ax_bg)
    ax.tick_params(colors=mc, labelsize=8)
    for s in ax.spines.values():
        s.set_color("#1e3555")
    ax.yaxis.label.set_color(mc)
    ax.xaxis.label.set_color(mc)
    ax.title.set_color(tc)

# ── Chart 1: LST Histogram ──
flat = lst_raw[np.isfinite(lst_raw)].flatten()
n, bins, patches = axes[0].hist(flat, bins=60, edgecolor="none")
# Color by temperature
bin_centers = 0.5 * (bins[:-1] + bins[1:])
norm_vals   = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min() + 1e-9)
cmap_inst   = matplotlib.colormaps[lst_cmap]
for patch, nv in zip(patches, norm_vals):
    patch.set_facecolor(cmap_inst(nv))
axes[0].axvline(avg_lst, color="#ffffff", lw=1.5, linestyle="--",
                label=f"Mean {avg_lst:.1f}°C")
axes[0].axvline(max_lst, color="#ff5722", lw=1, linestyle=":",
                label=f"Max {max_lst:.1f}°C")
axes[0].axvline(min_lst, color="#00bcd4", lw=1, linestyle=":",
                label=f"Min {min_lst:.1f}°C")
axes[0].set_xlabel("Temperature (°C)", fontsize=8)
axes[0].set_ylabel("Pixel Count", fontsize=8)
axes[0].set_title("LST Distribution", fontsize=9, fontweight="bold")
axes[0].legend(fontsize=7, labelcolor=tc, framealpha=0)

# ── Chart 2: Land Cover Donut ──
sizes  = [stats["buildings"], stats["vegetation"], stats["water"]]
colors = ["#dc3c3c", "#32b450", "#2882dc"]
labels = [f"Built-up\n{sizes[0]:.1f}%",
          f"Vegetation\n{sizes[1]:.1f}%",
          f"Water\n{sizes[2]:.1f}%"]
wedges, _ = axes[1].pie(
    sizes, colors=colors, startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=fig_bg, linewidth=2)
)
axes[1].set_title("Land Cover", fontsize=9, fontweight="bold")
axes[1].legend(wedges, labels, loc="center left", bbox_to_anchor=(0.85, 0.5),
               fontsize=7, labelcolor=tc, framealpha=0)

# ── Chart 3: Zonal Stats Bar ──
zonal = ss.zonal_rows
valid_z = [z for z in zonal if z["mean"] is not None]
if valid_z:
    z_names  = [z["class"] for z in valid_z]
    z_means  = [z["mean"]  for z in valid_z]
    z_stds   = [z["std"]   for z in valid_z]
    z_colors = {"Built-up":"#dc3c3c","Vegetation":"#32b450","Water":"#2882dc"}
    bar_cols  = [z_colors.get(n, "#888") for n in z_names]
    bars = axes[2].bar(z_names, z_means, color=bar_cols, edgecolor="none",
                       width=0.55, zorder=2)
    axes[2].errorbar(z_names, z_means, yerr=z_stds,
                     fmt="none", color="#ffffff", capsize=4, linewidth=1.2)
    axes[2].set_ylabel("Mean LST (°C)", fontsize=8)
    axes[2].set_title("Zonal Mean LST by Land Cover", fontsize=9, fontweight="bold")
    axes[2].tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, z_means):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}°", ha="center", va="bottom",
                     fontsize=7, color=tc)

plt.tight_layout(pad=1.2)
st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ── Zonal Statistics Table ────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="sec-label">📋 Zonal Statistics</div>', unsafe_allow_html=True)

def _fmt(v, unit="°C", prec=2):
    return f"{v:.{prec}f} {unit}" if v is not None else "—"

def _cls_color(row):
    if row["mean"] is None:
        return "mid"
    if row["mean"] >= avg_lst + std_lst:
        return "hot"
    elif row["mean"] <= avg_lst - std_lst:
        return "ok"
    else:
        return "mid"

rows_html = ""
for z in ss.zonal_rows:
    cc = _cls_color(z)
    rows_html += f"""
    <tr>
      <td>{z['class']}</td>
      <td>{z['pixels']:,}</td>
      <td class="{cc}">{_fmt(z['mean'])}</td>
      <td class="hot">{_fmt(z['max'])}</td>
      <td class="ok">{_fmt(z['min'])}</td>
      <td>{_fmt(z['std'])}</td>
    </tr>"""

st.markdown(f"""
<div class="info-card">
  <table class="zt">
    <thead><tr>
      <th>Land Cover</th>
      <th>Pixels</th>
      <th>Mean LST</th>
      <th>Max LST</th>
      <th>Min LST</th>
      <th>Std Dev</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)


# ── UHI Summary ──────────────────────────────────────────────────────────────
buildup_z = next((z for z in ss.zonal_rows if z["class"] == "Built-up"), None)
veg_z     = next((z for z in ss.zonal_rows if z["class"] == "Vegetation"), None)

if buildup_z and veg_z and buildup_z["mean"] and veg_z["mean"]:
    delta = buildup_z["mean"] - veg_z["mean"]
    st.markdown(f"""
    <div class="info-card">
      <div class="info-card-title">🏙️ UHI Summary</div>
      <p style="font-size:.82rem;color:#6b8cad;margin:0">
        Built-up พื้นที่มีอุณหภูมิเฉลี่ย
        <strong style="color:#ff5722">{buildup_z['mean']:.2f}°C</strong>
        สูงกว่าพื้นที่พืชพรรณ
        <strong style="color:#4caf50">{veg_z['mean']:.2f}°C</strong>
        อยู่ที่ <strong style="color:#ff8f00">{delta:+.2f}°C</strong>
        &nbsp;·&nbsp; UHI Intensity (Max−Min) = <strong style="color:#ff1744">{uhi_int:.2f}°C</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)


# ── ESG PDF Export ────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="sec-label">📄 ESG Report Export</div>', unsafe_allow_html=True)

exp_col, _ = st.columns([1, 3])
with exp_col:
    if st.button("📥 Export ESG PDF Report", use_container_width=True):
        with st.spinner("กำลังสร้าง PDF..."):
            pdf_bytes = generate_esg_pdf(
                avg_lst=avg_lst, max_lst=max_lst, min_lst=min_lst,
                uhi_int=uhi_int, std_lst=std_lst,
                seg_stats=stats, zonal_rows=ss.zonal_rows,
                gee_info=ss.gee_info,
                lst_rgb=ss.lst_rgb, seg_mask=ss.seg_mask, rgb_array=ss.rgb_array,
            )
        st.download_button(
            label="⬇️ ดาวน์โหลด ESG_Report.pdf",
            data=pdf_bytes,
            file_name=f"ESG_UHI_Report_{date.today().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.markdown(
            '<div class="alert-ok">✅ PDF พร้อมดาวน์โหลด — เหมาะสำหรับแนบรายงาน ESG</div>',
            unsafe_allow_html=True,
        )


# ── Technical Notes ───────────────────────────────────────────────────────────
with st.expander("📖 Technical Notes"):
    st.markdown(f"""
### 🌍 GEE LST Pipeline (Landsat C2 L2)
```
ST_B10 (DN) × 0.00341802 + 149.0  →  Brightness Temperature (K)
                           − 273.15  →  LST (°C)
QA_PIXEL: bit3 = Cloud, bit4 = Cloud Shadow  →  mask
Landsat 8 (LC08/C02/T1_L2) + Landsat 9 (LC09/C02/T1_L2)
.filterBounds(ROI) .filterDate() .filter(CLOUD_COVER < {cloud_pct}%)
.map(mask_clouds) .map(to_celsius) .median() .clip(ROI)
→ getDownloadURL (GeoTIFF, EPSG:4326, {gee_scale}m)
→ rasterio MemoryFile → numpy float32
```

### 🤖 Land Classification
```
Rule-based spectral thresholding on RGB bands:
  Vegetation : G > R+18 AND G > B+18 AND G > 60
  Water      : B > R+18 AND B > G+8  AND B > 55
  Built-up   : ทุกพิกเซลที่เหลือ (default class)
```

### 📂 File Handling
- รองรับ GeoTIFF สูงสุด **10 GB** (`.streamlit/config.toml`)
- Downsample อัตโนมัติ ≤ {MAX_PX}px ด้านยาวสุดเพื่อ display
- LST array จาก GEE resize ให้ตรงกับ RGB ก่อน overlay

### 🗺️ Layer Control Fix
- ใช้ `folium.FeatureGroup` ห่อ `ImageOverlay` ทุก layer
- `LayerControl` ต้อง `.add_to(m)` หลัง FeatureGroup ทั้งหมด
- `show=True/False` ตั้งใน `FeatureGroup` ไม่ใช่ `ImageOverlay`
    """)