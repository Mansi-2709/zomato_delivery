from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =============================================================================
# Zomato Delivery Intelligence — Streamlit UI only
# No ML training, model loading, or inference is implemented here.
# Keep your cleaned dataset beside this file as `zomato_cleaned.csv`.
# =============================================================================

st.set_page_config(
    page_title="Zomato | Delivery Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RED = "#E23744"
RED_DARK = "#B71C2C"
RED_LIGHT = "#FFF0F2"
RED_SOFT = "#FCE2E5"
INK = "#2B2225"
MUTED = "#81767A"
BORDER = "#F0D9DC"
WHITE = "#FFFFFF"
PALETTE = [RED, "#FF7E8B", "#FFB4BD", "#B71C2C", "#6B333B", "#FFD8DD", "#A99A9E", "#F5A0AA"]

# ------------------------------- Global theme -------------------------------
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root {{
        --z-red: {RED}; --z-red-dark: {RED_DARK}; --z-red-light: {RED_LIGHT};
        --z-ink: {INK}; --z-muted: {MUTED}; --z-border: {BORDER};
      }}
      html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; color: var(--z-ink); }}
      .stApp {{ background: linear-gradient(180deg, #FFF7F8 0, #FFFFFF 340px); }}
      [data-testid="stHeader"] {{ background: transparent; height: 0; }}
      [data-testid="stToolbar"], #MainMenu, footer {{ visibility: hidden; }}
      [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
      .block-container {{ max-width: 1520px; padding-top: 1.15rem; padding-bottom: 2.8rem; }}
      h1, h2, h3, h4 {{ font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; color: var(--z-ink); }}
      p, label, .stCaption {{ color: var(--z-muted); }}

      .topline {{ height: 5px; border-radius: 999px; background: linear-gradient(90deg, {RED_DARK}, {RED}, #FF8F9B); margin-bottom: 18px; }}
      .hero {{
        display: flex; align-items: center; justify-content: space-between; gap: 28px;
        padding: 24px 28px; margin: 0 0 18px 0; border-radius: 24px;
        background: linear-gradient(112deg, #FFFFFF 0%, #FFF2F4 78%, #FFE5E8 100%);
        border: 1px solid var(--z-border); box-shadow: 0 12px 32px rgba(142, 23, 41, .075);
      }}
      .brand-block {{ display:flex; align-items:center; gap:22px; min-width:0; }}
      .logo-wrap {{ background:#FFFFFF; border:1px solid #F4D9DD; border-radius:16px; padding:8px 13px 10px; box-shadow:0 5px 16px rgba(226,55,68,.10); flex-shrink:0; }}
      .logo-wordmark {{ color:{RED}; font-family:'Arial Black', 'Arial', sans-serif; font-size:2.45rem; font-weight:900; line-height:1; letter-spacing:-.105em; display:block; transform:scaleX(.98); }}
      .hero-kicker {{ color:var(--z-red); text-transform:uppercase; font-weight:800; font-size:.71rem; letter-spacing:.16em; margin:0 0 6px 0; }}
      .hero-title {{ margin:0; font-size:clamp(1.65rem, 2.7vw, 2.45rem); line-height:1.08; font-weight:800; }}
      .hero-subtitle {{ margin:8px 0 0; font-size:.94rem; max-width:760px; color:#766A6E; line-height:1.5; }}
      .hero-badge {{ color:{RED_DARK}; background:#FFFFFF; border:1px solid #F2CDD2; border-radius:999px; padding:10px 14px; font-size:.75rem; font-weight:800; letter-spacing:.055em; white-space:nowrap; }}
      .section-heading {{ display:flex; align-items:center; gap:10px; margin: 17px 0 11px; }}
      .section-heading .dot {{ width:9px; height:9px; border-radius:50%; background:var(--z-red); display:inline-block; box-shadow:0 0 0 4px #FDE7EA; }}
      .section-heading h3 {{ margin:0; font-size:1.12rem; font-weight:800; }}
      .section-note {{ color:var(--z-muted); font-size:.87rem; margin:0 0 13px; }}
      .metric-card {{ background:#FFFFFF; border:1px solid var(--z-border); border-radius:17px; padding:16px 17px; box-shadow:0 7px 20px rgba(58,25,32,.035); min-height:112px; }}
      .metric-label {{ font-size:.72rem; color:#897C81; font-weight:800; text-transform:uppercase; letter-spacing:.085em; }}
      .metric-value {{ margin-top:8px; color:var(--z-ink); font-family:'Manrope',sans-serif; font-size:1.75rem; line-height:1.1; font-weight:800; }}
      .metric-foot {{ color:#9A8D92; font-size:.78rem; margin-top:8px; }}
      .soft-card {{ background:#FFFFFF; border:1px solid var(--z-border); border-radius:18px; padding:18px 20px; box-shadow:0 8px 22px rgba(58,25,32,.035); }}
      .soft-card h4 {{ margin:0 0 7px; color:var(--z-ink); }} .soft-card p {{ margin:0; color:var(--z-muted); }}
      .pill {{ display:inline-block; padding:6px 10px; border-radius:999px; background:#FFE8EB; color:{RED_DARK}; font-size:.74rem; font-weight:800; letter-spacing:.04em; }}
      .stTabs [data-baseweb="tab-list"] {{ display:flex; gap:10px; padding:8px; background:#FFFFFF; border:1px solid var(--z-border); border-radius:17px; margin:0 0 22px; box-shadow:0 8px 24px rgba(58,25,32,.04); }}
      .stTabs [data-baseweb="tab"] {{ flex:1; justify-content:center; height:53px; padding:0 16px; border-radius:12px; color:#706267; font-weight:800; border:1px solid transparent; }}
      .stTabs [data-baseweb="tab"] p {{ font-size:.93rem; }}
      .stTabs [data-baseweb="tab"][aria-selected="true"] {{ background:{RED}; color:#FFFFFF; box-shadow:0 5px 14px rgba(226,55,68,.22); }}
      .stTabs [data-baseweb="tab"][aria-selected="true"] p {{ color:#FFFFFF !important; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background:transparent; }}
      .stButton > button, .stFormSubmitButton > button {{ border-radius:12px; border:1px solid {RED}; background:{RED}; color:#FFFFFF; font-weight:800; padding:.65rem 1.1rem; }}
      .stButton > button:hover, .stFormSubmitButton > button:hover {{ background:{RED_DARK}; border-color:{RED_DARK}; color:#FFFFFF; }}
      .stSelectbox [data-baseweb="select"] > div, .stMultiSelect [data-baseweb="select"] > div {{ border-radius:11px; border-color:#E8D6D9; }}
      [data-testid="stMetric"] {{ background:#FFFFFF; border:1px solid var(--z-border); border-radius:15px; padding:13px 15px; }}
      [data-testid="stMetricLabel"] {{ color:#897C81; }} [data-testid="stMetricValue"] {{ color:var(--z-ink); }}
      [data-testid="stDataFrame"] {{ border:1px solid var(--z-border); border-radius:13px; overflow:hidden; }}
      .prediction-placeholder {{ border:1px dashed #E8A7AF; background:linear-gradient(135deg,#FFFFFF,#FFF3F5); border-radius:20px; padding:22px; min-height:220px; }}
      .prediction-placeholder h3 {{ margin:12px 0 6px; color:{RED_DARK}; }}
      .prediction-value {{ font-family:'Manrope',sans-serif; font-size:2.75rem; font-weight:800; color:{RED}; line-height:1.1; }}
      .footer {{ border-top:1px solid var(--z-border); margin-top:28px; padding-top:14px; display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; color:#9A8D92; font-size:.79rem; }}
      @media (max-width: 760px) {{
        .hero {{ padding:18px; }} .brand-block {{ align-items:flex-start; gap:13px; }}
        .logo-wordmark {{ font-size:1.8rem; }} .hero-badge {{ display:none; }}
        .stTabs [data-baseweb="tab"] {{ padding:0 5px; height:auto; min-height:48px; }}
        .stTabs [data-baseweb="tab"] p {{ font-size:.75rem; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)
# ----------------------------------------------------------------------------------------

st.markdown(
        f"""
        <div class="topline"></div>
        <div class="hero">
          <div class="brand-block">
            <div class="logo-wrap"><span class="logo-wordmark">zomato</span></div>
            <div>
              <div class="hero-kicker">Delivery operations • Analytics workspace</div>
              <h1 class="hero-title">Delivery Intelligence</h1>
              <p class="hero-subtitle">Explore delivery insights, geographic clusters and prediction inputs in one workspace.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("The Zomato dataset was not found. No upload control is provided in this app.")
    st.markdown(
        "Place your cleaned CSV in the same folder as `app.py` and name it **`zomato_cleaned.csv`**. "
        "The app will load it automatically the next time it runs."
    )
    st.stop()



# The three tabs are in the main page and visible immediately below the header.
tab_eda, tab_cluster, tab_predict = st.tabs(
    ["📊  EDA & INSIGHTS", "📍  CLUSTERING", "⏱️  PREDICTIONS"]
)

# ================================= TAB 1 =====================================
with tab_eda:
    

# ================================= TAB 2 =====================================
with tab_cluster:
    
# ================================= TAB 3 =====================================
with tab_predict:
  
# -------------------------------- Footer -------------------------------------
st.markdown(
    f"""
    <div class="footer">
      <span><b style="color:{RED}">zomato</b> · Delivery Intelligence Portfolio</span>
      <span>EDA & Insights · Clustering · Prediction UI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

