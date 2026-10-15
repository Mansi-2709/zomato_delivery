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

# =============================================================================
# LOAD DATA
# =============================================================================
delivery_data = pd.read_csv('https://raw.githubusercontent.com/Mansi-2709/zomato_delivery/refs/heads/master/Zomato2.csv')

# =============================================================================
# PREPARE DAY-OF-WEEK LABELS
# =============================================================================

day_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

day_order = list(day_map.values())

delivery_data["Day_of_Week_Num"] = pd.to_numeric(
    delivery_data["Day_of_Week_Num"],
    errors="coerce"
).astype("Int64")

delivery_data["Day_of_Week"] = (
    delivery_data["Day_of_Week_Num"].map(day_map)
)

# Normalize traffic labels for consistent plot colors

traffic_colors = {
    "High": RED_DARK,
    "Medium": RED,
    "Low": "#FF9AA5",
    "Jam": "#7B1722"
}

# =============================================================================
# PLOTLY RED-AND-WHITE THEME
# =============================================================================

def style_chart(fig, title, height=370):

    fig.update_layout(
        template="plotly_white",
        title={
            "text": title,
            "x": 0.025,
            "xanchor": "left",
            "font": {
                "family": "Manrope, Arial, sans-serif",
                "size": 17,
                "color": INK
            }
        },
        font={
            "family": "DM Sans, Arial, sans-serif",
            "size": 12,
            "color": INK
        },
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        height=height,
        margin=dict(l=16, r=16, t=66, b=22),
        hoverlabel=dict(
            bgcolor=WHITE,
            font_size=12,
            font_family="DM Sans, Arial, sans-serif",
            font_color=INK,
            bordercolor=BORDER
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0)"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=BORDER,
        linewidth=1,
        ticks="outside",
        tickcolor=BORDER,
        tickfont=dict(size=11, color=MUTED),
        title_font=dict(size=12, color=INK),
        automargin=True
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F6E8EA",
        gridwidth=1,
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color=MUTED),
        title_font=dict(size=12, color=INK),
        automargin=True
    )

    return fig


# =============================================================================
# MAIN TABS — ALL VISIBLE ON THE FIRST PAGE
# =============================================================================

tab_eda, tab_clustering, tab_prediction = st.tabs([
    "📊  EDA & INSIGHTS",
    "📍  CLUSTERING INTELLIGENCE",
    "⏱️  DELIVERY PREDICTION"
])


# =============================================================================
# TAB 1 — EDA & GRAPHS
# =============================================================================

with tab_eda:

    st.html(
        f"""
        <div style="
            padding: 8px 2px 16px 2px;
        ">
            <div style="
                color: {RED};
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 5px;
            ">EXPLORE YOUR DATA</div>

            <div style="
                color: {INK};
                font-size: 25px;
                font-weight: 800;
                letter-spacing: -0.7px;
            ">Delivery Performance Overview</div>

            <div style="
                color: {MUTED};
                font-size: 14px;
                margin-top: 5px;
            ">Understand the relationship between distance, traffic, order timing, weather and delivery performance.</div>
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # GRAPH 1 — Distance vs Delivery Time
    # -------------------------------------------------------------------------

    st.markdown(
        f"""
        <div style="color:{RED_DARK}; font-size:12px;
                    font-weight:800; letter-spacing:1px;
                    margin:8px 0 10px 0;">
            DELIVERY DISTANCE & TIMING
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):

            fig_distance = px.scatter(
                delivery_data,
                x="distance_km",
                y="Time_taken (min)",
                color_discrete_map=traffic_colors,
                labels={
                    "distance_km": "Delivery distance (km)",
                    "Time_taken (min)": "Delivery time (min)"
                },
                title="Distance vs Delivery Time",
                opacity=0.72
            )

            fig_distance.update_traces(
                marker=dict(
                    size=7,
                    line=dict(width=0.5, color=WHITE)
                )
            )

            fig_distance = style_chart(
                fig_distance,
                "Distance vs Delivery Time",
                height=390
            )

            st.plotly_chart(
                fig_distance,
                use_container_width=True,
                config={"displayModeBar": False}
            )

            st.caption(
                "Insight: Examine whether longer delivery distances "
                "are associated with longer delivery times, and compare "
                "the patterns across traffic conditions."
            )

    # -------------------------------------------------------------------------
    # GRAPH 2 — Average Delivery Time by Order Hour
    # -------------------------------------------------------------------------

    with col2:
        with st.container(border=True):

            orderhour_df = (
                delivery_data
                .groupby("Time_Orderd_Hour", as_index=False)
                ["Time_taken (min)"]
                .mean()
                .sort_values("Time_Orderd_Hour")
            )

            fig_hour = px.line(
                orderhour_df,
                x="Time_Orderd_Hour",
                y="Time_taken (min)",
                markers=True,
                labels={
                    "Time_Orderd_Hour": "Order hour",
                    "Time_taken (min)": "Average delivery time (min)"
                },
                title="Delivery Time by Order Hour"
            )

            fig_hour.update_traces(
                line=dict(color=RED, width=3),
                marker=dict(
                    color=RED_DARK,
                    size=8,
                    line=dict(color=WHITE, width=2)
                ),
                hovertemplate=(
                    "Hour: %{x}<br>"
                    "Avg delivery time: %{y:.1f} min"
                    "<extra></extra>"
                )
            )

            fig_hour = style_chart(
                fig_hour,
                "Delivery Time by Order Hour",
                height=390
            )

            st.plotly_chart(
                fig_hour,
                use_container_width=True,
                config={"displayModeBar": False}
            )

            st.caption(
                "Insight: Identify periods when average delivery time "
                "rises or falls throughout the day."
            )

    # -------------------------------------------------------------------------
    # GRAPH 3 — Average Delivery Time by Vehicle Type
    # -------------------------------------------------------------------------

    st.markdown(
        f"""
        <div style="color:{RED_DARK}; font-size:12px;
                    font-weight:800; letter-spacing:1px;
                    margin:20px 0 10px 0;">
            OPERATING CONDITIONS
        </div>
        """,
        unsafe_allow_html=True
    )

    col3, col4 = st.columns(2, gap="large")

    with col3:
        with st.container(border=True):

            vehicle_df = (
                delivery_data
                .groupby("Type_of_vehicle", as_index=False)
                ["Time_taken (min)"]
                .mean()
            )

            fig_vehicle = px.line(
                vehicle_df,
                x="Type_of_vehicle",
                y="Time_taken (min)",
                markers= True,
                labels={
                    "Type_of_vehicle": "Vehicle type",
                    "Time_taken (min)": "Average delivery time (min)"
                },
                title="Average Delivery Time by Vehicle"
            )

            fig_vehicle.update_traces(
                line=dict(color=RED, width=3),
                marker=dict(
                    color=RED_DARK,
                    size=8,
                    line=dict(color=WHITE, width=2)
                ),
                hovertemplate=(
                    "Hour: %{x}<br>"
                    "Avg delivery time: %{y:.1f} min"
                    "<extra></extra>"
                )
            )

            fig_vehicle = style_chart(
                fig_vehicle,
                "Average Delivery Time by Vehicle",
                height=365
            )

            fig_vehicle.update_layout(showlegend=False)

            st.plotly_chart(
                fig_vehicle,
                use_container_width=True,
                config={"displayModeBar": False}
            )

            st.caption(
                "Compare average delivery time across vehicle categories."
            )

    # -------------------------------------------------------------------------
    # GRAPH 4 — Average Delivery Time by Weather
    # -------------------------------------------------------------------------

    with col4:
        with st.container(border=True):

            weather_df = (
                delivery_data
                .groupby("Weather_conditions", as_index=False)
                ["Time_taken (min)"]
                .mean()
            )

            fig_weather = px.line(
                weather_df,
                x="Weather_conditions",
                y="Time_taken (min)",
                markers=True,
                labels={
                    "Weather_conditions": "Weather",
                    "Time_taken (min)": "Average delivery time (min)"
                },
                title="Average Delivery Time by Weather"
            )

            fig_weather.update_traces(
                line=dict(color=RED, width=3),
                marker=dict(
                    color=RED_DARK,
                    size=8,
                    line=dict(color=WHITE, width=2)
                ),
                hovertemplate=(
                    "Hour: %{x}<br>"
                    "Avg delivery time: %{y:.1f} min"
                    "<extra></extra>"
                )
            )

            fig_weather = style_chart(
                fig_weather,
                "Average Delivery Time by Weather",
                height=365
            )

            fig_weather.update_layout(showlegend=False)

            st.plotly_chart(
                fig_weather,
                use_container_width=True,
                config={"displayModeBar": False}
            )

            st.caption(
                "Compare average delivery time under different weather "
                "conditions."
            )

    # -------------------------------------------------------------------------
    # GRAPH 5 — Orders by Day of Week
    # -------------------------------------------------------------------------

    st.markdown(
        f"""
        <div style="color:{RED_DARK}; font-size:12px;
                    font-weight:800; letter-spacing:1px;
                    margin:20px 0 10px 0;">
            WEEKLY ORDER PATTERNS
        </div>
        """,
        unsafe_allow_html=True
    )

    day_col, day_insight_col = st.columns([1.5, 1], gap="large")

    with day_col:
        with st.container(border=True):

            day_df = delivery_data.dropna(
                subset=["Day_of_Week"]
            )

            fig_day = px.histogram(
                day_df,
                x="Day_of_Week",
                category_orders={
                    "Day_of_Week": day_order
                },
                color_discrete_sequence=[RED],
                labels={
                    "Day_of_Week": "Day of week",
                    "count": "Number of orders"
                },
                title="Order Distribution by Day"
            )

            fig_day.update_traces(
                marker_line_color=WHITE,
                marker_line_width=1.5,
                hovertemplate=(
                    "%{x}<br>"
                    "Records: %{y}"
                    "<extra></extra>"
                )
            )

            fig_day = style_chart(
                fig_day,
                "Order Distribution by Day",
                height=350
            )

            # Increase the gap between bars to make them slimmer
            fig_day.update_layout(
                bargap=0.55,
                showlegend=False,
                xaxis=dict(
                    categoryorder="array",
                    categoryarray=day_order
                )
            )

            st.plotly_chart(
                fig_day,
                use_container_width=True,
                config={"displayModeBar": False}
            )

    with day_insight_col:
        with st.container(border=True):

            day_counts = (
                delivery_data["Day_of_Week"]
                .value_counts()
                .reindex(day_order, fill_value=0)
            )

            total_day_records = int(day_counts.sum())

            busiest_day = (
                day_counts.idxmax()
                if total_day_records > 0
                else "N/A"
            )

            busiest_count = (
                int(day_counts.max())
                if total_day_records > 0
                else 0
            )

            busiest_pct = (
                busiest_count / total_day_records * 100
                if total_day_records > 0
                else 0
            )

            st.markdown(
                f"""
                <div style="padding:8px 4px 14px 4px;">
                    <div style="
                        color:{MUTED};
                        font-size:11px;
                        font-weight:800;
                        letter-spacing:1.4px;
                        text-transform:uppercase;
                    ">WEEKLY SNAPSHOT</div>

                    <div style="
                        color:{INK};
                        font-size:24px;
                        font-weight:800;
                        margin-top:10px;
                    ">{busiest_day}</div>

                    <div style="
                        color:{RED};
                        font-size:13px;
                        font-weight:700;
                        margin-top:3px;
                    ">Highest order count</div>

                    <div style="
                        background:{RED_LIGHT};
                        border-radius:13px;
                        padding:15px;
                        margin-top:18px;
                    ">
                        <div style="color:{MUTED};font-size:12px;">
                            Records on busiest day
                        </div>
                        <div style="
                            color:{RED_DARK};
                            font-size:26px;
                            font-weight:800;
                            margin-top:3px;
                        ">{busiest_count:,}</div>

                        <div style="color:{MUTED};font-size:12px;">
                            {busiest_pct:.1f}% of weekday-labelled records
                        </div>
                    </div>

                    <div style="
                        color:{MUTED};
                        font-size:13px;
                        line-height:1.7;
                        margin-top:15px;
                    ">Use this comparison to investigate which days record the highest order volume. The busiest day is based on record counts, not delivery speed.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        f"""
        <div style="
            border-top:1px solid {BORDER};
            margin-top:24px;
            padding-top:12px;
            color:{MUTED};
            font-size:12px;
        ">
            <b style="color:{RED}">Analysis note:</b>
            These charts show descriptive patterns. Differences in average
            delivery time do not, by themselves, establish causation.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 2 — CLUSTERING
# Keep/move your existing clustering map and cluster profile code here.
# =============================================================================

with tab_clustering:

    st.subheader("Clustering Intelligence")
    st.caption(
        "Move your existing geographic clustering maps, cluster metrics, "
        "and cluster profile tables into this tab."
    )

    # Paste your existing clustering map and cluster profile code here.


# =============================================================================
# TAB 3 — PREDICTION
# Keep/move your existing prediction UI code here.
# No model training or inference is included in this UI structure.
# =============================================================================

with tab_prediction:

    st.subheader("Delivery Time Prediction")
    st.caption(
        "Move your existing delivery prediction inputs and prediction-result "
        "layout into this tab."
    )

    # Paste your existing prediction form and result UI here.


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    f"""
    <div class="footer">
      <span><b style="color:{RED}">zomato</b> · Delivery Intelligence Portfolio</span>
      <span>EDA & Insights · Clustering · Prediction UI</span>
    </div>
    """,
    unsafe_allow_html=True,
)
