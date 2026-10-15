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

# -------------------------------- Helpers ------------------------------------
def pick_col(frame: pd.DataFrame, *candidates: str):
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt(value, decimals=1, suffix=""):
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


def add_metric_row(items):
    cols = st.columns(len(items), gap="medium")
    for col, (label, value, detail) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{value}</div>
                  <div class="metric-foot">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def section_title(title, note=None):
    st.markdown(
        f'<div class="section-heading"><span class="dot"></span><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)


def polish_plot(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=52, b=12),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="DM Sans, sans-serif", color=INK),
        title_font=dict(family="Manrope, sans-serif", size=16, color=INK),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#F3E5E7", zeroline=False)
    return fig


def dominant_value(series):
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return "Not available"
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else "Not available"


def city_filter(frame, city_column, key):
    if not city_column:
        return frame.copy()
    cities = sorted(frame[city_column].dropna().astype(str).unique().tolist())
    options = ["All cities"] + cities
    chosen = st.selectbox("City", options, key=key)
    if chosen == "All cities":
        return frame.copy()
    return frame[frame[city_column].astype(str) == chosen].copy()

# ------------------------------ Load data ------------------------------------
# No uploader: the app reads a known local file beside app.py.
APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    APP_DIR / "zomato_cleaned.csv",
    APP_DIR / "zomato_delivery_cleaned.csv",
    APP_DIR / "delivery_data.csv",
    APP_DIR / "zomato.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)

if DATA_PATH is None:
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

try:
    df = pd.read_csv(DATA_PATH)
except Exception as exc:
    st.error(f"Could not read `{DATA_PATH.name}`: {exc}")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]

# Detect column names from the provided Zomato dataset.
cluster_col = pick_col(df, "operational_cluster", "geo_cluster", "cluster", "hdbscan_cluster")
city_col = pick_col(df, "City", "city")
time_col = pick_col(df, "Time_taken (min)", "Time_taken", "delivery_time", "time_taken_min")
distance_col = pick_col(df, "delivery_distance_km", "distance_km", "distance", "Distance")
rating_col = pick_col(df, "Delivery_person_Ratings", "delivery_person_ratings", "rating")
age_col = pick_col(df, "Delivery_person_Age", "delivery_person_age")
traffic_col = pick_col(df, "Road_traffic_density", "road_traffic_density", "traffic_density")
weather_col = pick_col(df, "Weather_conditions", "weather_conditions", "weather")
vehicle_condition_col = pick_col(df, "Vehicle_condition", "vehicle_condition")
multiple_col = pick_col(df, "multiple_deliveries", "Multiple_deliveries")
order_type_col = pick_col(df, "Type_of_order", "type_of_order", "order_type")
vehicle_type_col = pick_col(df, "Type_of_vehicle", "type_of_vehicle", "vehicle_type")
festival_col = pick_col(df, "Festival", "festival")
order_date_col = pick_col(df, "Order_Date", "order_date", "date")
order_time_col = pick_col(df, "Time_Orderd_Hour", "Time_Orderd", "Time_Ordered", "order_hour")
restaurant_lat_col = pick_col(df, "Restaurant_latitude", "restaurant_latitude")
restaurant_lon_col = pick_col(df, "Restaurant_longitude", "restaurant_longitude")
delivery_lat_col = pick_col(df, "Delivery_location_latitude", "Delivery_latitude", "delivery_location_latitude")
delivery_lon_col = pick_col(df, "Delivery_location_longitude", "Delivery_longitude", "delivery_location_longitude")

for col in [time_col, distance_col, rating_col, age_col, restaurant_lat_col, restaurant_lon_col, delivery_lat_col, delivery_lon_col]:
    if col:
        df[col] = as_numeric(df[col])

# ------------------------------- Visible header -----------------------------
st.markdown('<div class="topline"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
      <div class="brand-block">
        <div class="logo-wrap" aria-label="Zomato wordmark"><span class="logo-wordmark">zomato</span></div>
        <div>
          <div class="hero-kicker">Delivery operations • Machine learning portfolio</div>
          <h1 class="hero-title">Delivery Intelligence</h1>
          <p class="hero-subtitle">Track delivery performance, discover geographic hotspots, compare operational clusters, and prepare delivery-time predictions.</p>
        </div>
      </div>
      <div class="hero-badge">EDA&nbsp;&nbsp; · &nbsp;&nbsp;CLUSTERS&nbsp;&nbsp; · &nbsp;&nbsp;PREDICTIONS</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# The three tabs are in the main page and visible immediately below the header.
tab_eda, tab_cluster, tab_predict = st.tabs(
    ["📊  EDA & INSIGHTS", "📍  CLUSTERING", "⏱️  PREDICTIONS"]
)

# ================================= TAB 1 =====================================
with tab_eda:
    filtered = city_filter(df, city_col, "eda_city_filter")
    section_title("Delivery performance overview", "A business-focused view of order volume, duration, distance and operating conditions.")

    avg_time = filtered[time_col].mean() if time_col else np.nan
    avg_dist = filtered[distance_col].mean() if distance_col else np.nan
    avg_rating = filtered[rating_col].mean() if rating_col else np.nan
    add_metric_row([
        ("Total orders", f"{len(filtered):,}", "Records in selected scope"),
        ("Avg. delivery time", fmt(avg_time, 1, " min"), "Mean of available target values"),
        ("Avg. distance", fmt(avg_dist, 2, " km"), "Uses your engineered distance column" if distance_col else "Distance column not detected"),
        ("Avg. rider rating", fmt(avg_rating, 2, " / 5"), "Mean of available ratings"),
    ])

    left, right = st.columns(2, gap="large")
    with left:
        if time_col and filtered[time_col].notna().any():
            fig = px.histogram(
                filtered.dropna(subset=[time_col]), x=time_col, nbins=28,
                title="How long do deliveries take?", color_discrete_sequence=[RED],
                labels={time_col: "Delivery time (minutes)"},
            )
            fig.update_traces(marker_line_color="white", marker_line_width=1)
            st.plotly_chart(polish_plot(fig), use_container_width=True)
        else:
            st.info("A numeric `Time_taken (min)` column is needed for this chart.")

    with right:
        if traffic_col:
            temp = filtered.copy()
            temp[traffic_col] = temp[traffic_col].astype("string").str.strip().str.title().fillna("Missing")
            if time_col:
                summary = temp.groupby(traffic_col, dropna=False)[time_col].mean().reset_index()
                summary.columns = ["Traffic density", "Average delivery time (min)"]
                fig = px.bar(summary, x="Traffic density", y="Average delivery time (min)",
                             title="Delivery time by traffic density", color="Traffic density",
                             color_discrete_sequence=PALETTE, text_auto=".1f")
            else:
                summary = temp[traffic_col].value_counts(dropna=False).reset_index()
                summary.columns = ["Traffic density", "Orders"]
                fig = px.bar(summary, x="Traffic density", y="Orders", title="Orders by traffic density",
                             color_discrete_sequence=[RED], text_auto=True)
            st.plotly_chart(polish_plot(fig), use_container_width=True)
        else:
            st.info("No road-traffic column detected.")

    left, right = st.columns(2, gap="large")
    with left:
        if weather_col:
            weather_counts = filtered[weather_col].fillna("Missing").astype(str).str.strip().str.title().value_counts().reset_index()
            weather_counts.columns = ["Weather", "Orders"]
            fig = px.bar(weather_counts, x="Weather", y="Orders", title="Order volume by weather",
                         color="Weather", color_discrete_sequence=PALETTE, text_auto=True)
            st.plotly_chart(polish_plot(fig), use_container_width=True)
        else:
            st.info("No weather column detected.")

    with right:
        if city_col and time_col:
            city_time = (
                filtered.dropna(subset=[city_col, time_col])
                .groupby(city_col)[time_col].mean().sort_values(ascending=False).head(12)
                .sort_values().reset_index()
            )
            city_time.columns = ["City", "Average delivery time (min)"]
            fig = px.bar(city_time, x="Average delivery time (min)", y="City", orientation="h",
                         title="Cities with longest average delivery time",
                         color_discrete_sequence=[RED], text_auto=".1f")
            st.plotly_chart(polish_plot(fig, 390), use_container_width=True)
        elif distance_col and time_col:
            scatter = filtered.dropna(subset=[distance_col, time_col])
            if len(scatter) > 5000:
                scatter = scatter.sample(5000, random_state=42)
            fig = px.scatter(scatter, x=distance_col, y=time_col, title="Distance vs delivery time",
                             labels={distance_col: "Distance (km)", time_col: "Delivery time (min)"},
                             opacity=.55, color_discrete_sequence=[RED])
            st.plotly_chart(polish_plot(fig), use_container_width=True)
        else:
            st.info("No city/time or distance/time pairing was detected.")

    if distance_col and time_col:
        section_title("Distance and delivery time", "Use this relationship for exploration; it does not by itself establish causation.")
        scatter = filtered.dropna(subset=[distance_col, time_col])
        if len(scatter) > 6500:
            scatter = scatter.sample(6500, random_state=42)
        fig = px.scatter(
            scatter, x=distance_col, y=time_col,
            color=traffic_col if traffic_col else None,
            title="Delivery time against distance",
            labels={distance_col: "Distance (km)", time_col: "Delivery time (minutes)"},
            color_discrete_sequence=PALETTE, opacity=.62,
            hover_data=[c for c in [city_col, weather_col, multiple_col] if c],
        )
        st.plotly_chart(polish_plot(fig, 390), use_container_width=True)

# ================================= TAB 2 =====================================
with tab_cluster:
    filtered = city_filter(df, city_col, "cluster_city_filter")
    section_title("Clustering intelligence", "Explore geographic concentration and compare the delivery conditions within each cluster.")

    if cluster_col is None:
        st.warning("No cluster-label column found. Expected `operational_cluster`, `geo_cluster`, or `cluster`.")
    else:
        labels = filtered[cluster_col].dropna()
        unique_labels = labels.unique().tolist()
        meaningful_labels = [x for x in unique_labels if str(x) != "-1"]
        noise_pct = labels.astype(str).eq("-1").mean() * 100 if len(labels) else 0
        largest = int(filtered[cluster_col].value_counts().max()) if not filtered.empty and not filtered[cluster_col].dropna().empty else 0
        add_metric_row([
            ("Clusters", f"{len(meaningful_labels):,}", f"From `{cluster_col}`; noise excluded"),
            ("Noise / unassigned", fmt(noise_pct, 1, "%"), "Rows with label -1, if used"),
            ("Largest cluster", f"{largest:,}", "Records in the largest group"),
            ("Records in view", f"{len(filtered):,}", "Selected city scope"),
        ])

        map_options = []
        if delivery_lat_col and delivery_lon_col:
            map_options.append("Delivery locations")
        if restaurant_lat_col and restaurant_lon_col:
            map_options.append("Restaurant locations")

        map_left, map_right = st.columns([1.45, 1], gap="large")
        with map_left:
            section_title("Geographic cluster map", "Points are sampled where necessary to keep the map responsive.")
            if not map_options:
                st.warning("No complete latitude/longitude pair was detected for restaurants or delivery locations.")
            else:
                map_mode = st.radio("Map points", map_options, horizontal=True, key="map_points_choice")
                if map_mode == "Delivery locations":
                    lat_col, lon_col = delivery_lat_col, delivery_lon_col
                else:
                    lat_col, lon_col = restaurant_lat_col, restaurant_lon_col

                map_df = filtered.dropna(subset=[lat_col, lon_col, cluster_col]).copy()
                map_df = map_df[map_df[lat_col].between(-90, 90) & map_df[lon_col].between(-180, 180)]
                original_map_n = len(map_df)
                if original_map_n > 8500:
                    map_df = map_df.sample(8500, random_state=42)
                map_df["Cluster"] = map_df[cluster_col].astype(str)
                hover_cols = [c for c in [city_col, time_col, distance_col, traffic_col, weather_col] if c]
                fig = px.scatter_mapbox(
                    map_df, lat=lat_col, lon=lon_col, color="Cluster", hover_data=hover_cols,
                    zoom=3.2, height=515, mapbox_style="open-street-map",
                    title=f"{map_mode} by cluster", color_discrete_sequence=PALETTE,
                )
                fig.update_traces(marker={"size": 7, "opacity": .72})
                fig.update_layout(
                    margin=dict(l=0, r=0, t=45, b=0),
                    paper_bgcolor="rgba(255,255,255,0)",
                    font=dict(family="DM Sans, sans-serif", color=INK),
                    legend_title_text="Cluster",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Showing {len(map_df):,} of {original_map_n:,} valid locations.")

        with map_right:
            section_title("Cluster explorer")
            cluster_options = sorted(filtered[cluster_col].dropna().unique().tolist(), key=lambda x: str(x))
            if cluster_options:
                selected_cluster = st.selectbox(
                    "Select a cluster", cluster_options,
                    format_func=lambda value: f"Cluster {value}", key="cluster_detail_select",
                )
                selected = filtered[filtered[cluster_col] == selected_cluster]
                selected_time = selected[time_col].mean() if time_col else np.nan
                selected_distance = selected[distance_col].mean() if distance_col else np.nan
                selected_rating = selected[rating_col].mean() if rating_col else np.nan
                st.markdown(
                    f"""
                    <div class="soft-card">
                      <span class="pill">CLUSTER {selected_cluster}</span>
                      <h4 style="margin-top:13px;font-size:1.45rem">{len(selected):,} records</h4>
                      <p>Average delivery time: <b>{fmt(selected_time, 1, ' min')}</b></p>
                      <p style="margin-top:7px">Average distance: <b>{fmt(selected_distance, 2, ' km')}</b></p>
                      <p style="margin-top:7px">Average rider rating: <b>{fmt(selected_rating, 2)}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if traffic_col:
                    traffic_mix = selected[traffic_col].fillna("Missing").astype(str).str.strip().str.title().value_counts().reset_index()
                    traffic_mix.columns = ["Traffic", "Records"]
                    fig = px.pie(traffic_mix, names="Traffic", values="Records", hole=.62,
                                 title="Traffic composition", color_discrete_sequence=PALETTE)
                    fig.update_layout(margin=dict(l=6, r=6, t=48, b=6), paper_bgcolor="rgba(255,255,255,0)")
                    st.plotly_chart(fig, use_container_width=True)
                if weather_col:
                    weather_mix = selected[weather_col].fillna("Missing").astype(str).str.strip().str.title().value_counts().reset_index()
                    weather_mix.columns = ["Weather", "Records"]
                    st.markdown("**Weather composition**")
                    st.dataframe(weather_mix, use_container_width=True, hide_index=True)

        section_title("Cluster profile", "Summarized operational characteristics help interpret cluster labels in business terms.")
        aggregations = {"total_records": (cluster_col, "size")}
        if distance_col:
            aggregations["avg_distance_km"] = (distance_col, "mean")
        if time_col:
            aggregations["avg_delivery_time_min"] = (time_col, "mean")
        if rating_col:
            aggregations["avg_rider_rating"] = (rating_col, "mean")
        profile = filtered.groupby(cluster_col, dropna=False).agg(**aggregations).reset_index()
        for column, output_name in [
            (traffic_col, "dominant_traffic"), (weather_col, "dominant_weather"),
            (vehicle_condition_col, "dominant_vehicle_condition"), (multiple_col, "dominant_multiple_deliveries"),
            (city_col, "dominant_city"),
        ]:
            if column:
                modes = filtered.groupby(cluster_col, dropna=False)[column].agg(dominant_value).rename(output_name).reset_index()
                profile = profile.merge(modes, on=cluster_col, how="left")
        numeric_cols = profile.select_dtypes(include="number").columns
        profile[numeric_cols] = profile[numeric_cols].round(2)
        st.dataframe(profile, use_container_width=True, hide_index=True)
        st.download_button(
            "Download cluster profile", data=profile.to_csv(index=False).encode("utf-8"),
            file_name="zomato_cluster_profile.csv", mime="text/csv",
        )

# ================================= TAB 3 =====================================
with tab_predict:
    section_title("Delivery-time prediction", "Prediction form layout is ready for your saved model pipeline to be connected later.")
    st.markdown(
        """
        <div class="soft-card" style="margin-bottom:17px">
          <span class="pill">PREDICTION UI</span>
          <h4 style="margin-top:12px">Prepare a delivery scenario</h4>
          <p>This page captures inputs only. It does not train a model or generate a predicted delivery time until your saved preprocessing pipeline and regressor are connected.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cities = sorted(df[city_col].dropna().astype(str).unique().tolist()) if city_col else ["Not specified"]
    weather_values = sorted(df[weather_col].dropna().astype(str).unique().tolist()) if weather_col else ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]
    traffic_values = sorted(df[traffic_col].dropna().astype(str).unique().tolist()) if traffic_col else ["Low", "Medium", "High", "Jam"]
    order_values = sorted(df[order_type_col].dropna().astype(str).unique().tolist()) if order_type_col else ["Snack", "Meal", "Drinks", "Buffet"]
    vehicle_values = sorted(df[vehicle_type_col].dropna().astype(str).unique().tolist()) if vehicle_type_col else ["motorcycle", "scooter", "electric_scooter"]
    festival_values = sorted(df[festival_col].dropna().astype(str).unique().tolist()) if festival_col else ["No", "Yes"]

    form_col, output_col = st.columns([1.45, 1], gap="large")
    with form_col:
        with st.form("delivery_prediction_form", border=True):
            st.markdown("#### Order and route")
            a, b = st.columns(2)
            with a:
                input_city = st.selectbox("City", cities)
                input_order_type = st.selectbox("Order type", order_values)
                input_weather = st.selectbox("Weather conditions", weather_values)
                input_traffic = st.selectbox("Road traffic density", traffic_values)
            with b:
                input_vehicle_type = st.selectbox("Vehicle type", vehicle_values)
                input_festival = st.selectbox("Festival", festival_values)
                input_vehicle_condition = st.selectbox("Vehicle condition", [0, 1, 2])
                input_multiple = st.selectbox("Multiple deliveries", [0, 1, 2, 3])

            st.markdown("#### Delivery partner and time")
            a, b, c = st.columns(3)
            with a:
                input_age = st.slider("Delivery person age", 18, 60, 30)
            with b:
                input_rating = st.slider("Delivery person rating", 1.0, 5.0, 4.6, 0.1)
            with c:
                input_hour = st.slider("Order hour", 0, 23, 19)

            st.markdown("#### Coordinates")
            a, b, c, d = st.columns(4)
            with a:
                input_rest_lat = st.number_input("Restaurant lat", value=28.6139, format="%.6f")
            with b:
                input_rest_lon = st.number_input("Restaurant lon", value=77.2090, format="%.6f")
            with c:
                input_delivery_lat = st.number_input("Delivery lat", value=28.6200, format="%.6f")
            with d:
                input_delivery_lon = st.number_input("Delivery lon", value=77.2150, format="%.6f")

            submitted = st.form_submit_button("Prepare prediction inputs", use_container_width=True)

    with output_col:
        st.markdown(
            """
            <div class="prediction-placeholder">
              <span class="pill">MODEL NOT CONNECTED</span>
              <h3>Estimated delivery time</h3>
              <div class="prediction-value">— min</div>
              <p style="margin-top:9px">Your trained regressor and preprocessing pipeline will be connected here later. No estimate is fabricated by this UI.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div class="soft-card" style="margin-top:14px">
          <h4>Next integration step</h4>
          <p>Load your saved model and preprocessing pipeline, apply the same feature engineering used during training, then display the returned minutes in this panel.</p>
        </div>
        """, unsafe_allow_html=True)

    if submitted:
        inputs = {
            "City": input_city,
            "Type_of_order": input_order_type,
            "Type_of_vehicle": input_vehicle_type,
            "Weather_conditions": input_weather,
            "Road_traffic_density": input_traffic,
            "Festival": input_festival,
            "Vehicle_condition": input_vehicle_condition,
            "multiple_deliveries": input_multiple,
            "Time_Orderd_Hour": input_hour,
            "Delivery_person_Age": input_age,
            "Delivery_person_Ratings": input_rating,
            "Restaurant_latitude": input_rest_lat,
            "Restaurant_longitude": input_rest_lon,
            "Delivery_location_latitude": input_delivery_lat,
            "Delivery_location_longitude": input_delivery_lon,
        }
        st.session_state["prediction_inputs"] = inputs
        st.success("Inputs captured. The ML model is not connected, so no delivery-time prediction has been generated.")
        with st.expander("View captured inputs"):
            st.json(inputs)

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

