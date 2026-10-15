from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Zomato Delivery Intelligence — UI skeleton only
# No model training, loading, or inference is implemented in this file.
# Put your cleaned CSV beside this file, or upload it through the sidebar.
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Zomato | Delivery Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ZOMATO_RED = "#E23744"
ZOMATO_RED_DARK = "#B91F2B"
PALE_RED = "#FFF5F6"
MID_RED = "#FF7E8B"
LIGHT_RED = "#FFDBE0"
INK = "#282326"
MUTED = "#777078"
BORDER = "#F1DADD"
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Zomato.svg/256px-Zomato.svg.png"

# ----------------------------- Global styling -------------------------------
st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root {{
        --z-red: {ZOMATO_RED};
        --z-red-dark: {ZOMATO_RED_DARK};
        --z-pale: {PALE_RED};
        --z-ink: {INK};
        --z-muted: {MUTED};
        --z-border: {BORDER};
      }}
      html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; color: var(--z-ink); }}
      .stApp {{ background: linear-gradient(180deg, #fff9fa 0%, #ffffff 340px); }}
      [data-testid="stHeader"] {{ background: rgba(255,255,255,0); }}
      [data-testid="stToolbar"] {{ right: 1rem; }}
      #MainMenu {{ visibility: hidden; }}
      footer {{ visibility: hidden; }}
      .block-container {{ max-width: 1500px; padding-top: 1.1rem; padding-bottom: 3rem; }}
      [data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid var(--z-border); }}
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: var(--z-red-dark); }}
      h1, h2, h3 {{ font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; color: var(--z-ink); }}
      h1 {{ font-weight: 800 !important; }}
      p, label, .stCaption {{ color: var(--z-muted); }}
      .hero {{
        display: flex; align-items: center; justify-content: space-between; gap: 24px;
        padding: 28px 30px; margin: 0 0 22px 0; border-radius: 24px;
        background: linear-gradient(110deg, #ffffff 0%, #fff5f6 100%);
        border: 1px solid var(--z-border); box-shadow: 0 12px 30px rgba(163, 32, 45, .06);
      }}
      .hero-kicker {{ color: var(--z-red); text-transform: uppercase; font-weight: 800; font-size: .76rem; letter-spacing: .15em; margin-bottom: 8px; }}
      .hero-title {{ margin: 0; font-size: clamp(1.8rem, 3vw, 2.7rem); line-height: 1.08; color: var(--z-ink); font-family: 'Manrope', sans-serif; font-weight: 800; }}
      .hero-subtitle {{ margin: 10px 0 0 0; font-size: 1rem; max-width: 760px; color: #6f666c; line-height: 1.55; }}
      .hero-logo {{ width: 82px; height: 82px; object-fit: contain; border-radius: 18px; box-shadow: 0 8px 22px rgba(226,55,68,.14); }}
      .brand-fallback {{ background: var(--z-red); color: white; border-radius: 18px; width: 82px; height: 82px; display:flex; justify-content:center; align-items:center; font-family:'Manrope',sans-serif; font-weight:800; font-size:1rem; letter-spacing:-.07em; }}
      .section-heading {{ display:flex; align-items:center; gap:10px; margin: 18px 0 12px 0; }}
      .section-heading .dot {{ width:9px; height:9px; border-radius:50%; background: var(--z-red); display:inline-block; }}
      .section-heading h3 {{ margin:0; font-size:1.12rem; font-weight:800; }}
      .section-note {{ color: var(--z-muted); font-size:.9rem; margin-top:-4px; margin-bottom:14px; }}
      .metric-card {{ background:#fff; border:1px solid var(--z-border); border-radius:18px; padding:15px 17px; box-shadow:0 7px 20px rgba(58,25,32,.035); min-height:108px; }}
      .metric-label {{ font-size:.79rem; color:#81767c; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
      .metric-value {{ margin-top:8px; color:var(--z-ink); font-family:'Manrope',sans-serif; font-size:1.75rem; line-height:1.1; font-weight:800; }}
      .metric-accent {{ color:var(--z-red); }}
      .soft-card {{ background:#fff; border:1px solid var(--z-border); border-radius:20px; padding:18px 20px; box-shadow:0 8px 22px rgba(58,25,32,.035); }}
      .soft-card h4 {{ margin:0 0 6px 0; font-family:'Manrope',sans-serif; color:var(--z-ink); }}
      .soft-card p {{ margin:0; color:var(--z-muted); }}
      .pill {{ display:inline-block; padding:6px 10px; border-radius:999px; background:#ffe9ec; color:var(--z-red-dark); font-size:.78rem; font-weight:800; }}
      .stTabs [data-baseweb="tab-list"] {{ gap:8px; padding:8px; background:#fff; border:1px solid var(--z-border); border-radius:16px; margin-bottom:18px; }}
      .stTabs [data-baseweb="tab"] {{ height:48px; padding:0 20px; border-radius:12px; color:#6e6268; font-weight:700; }}
      .stTabs [data-baseweb="tab"][aria-selected="true"] {{ background:var(--z-red); color:#fff; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background:transparent; }}
      .stButton > button, .stFormSubmitButton > button {{ border-radius:12px; border:1px solid var(--z-red); background:var(--z-red); color:white; font-weight:800; padding:.62rem 1.1rem; transition:all .15s ease; }}
      .stButton > button:hover, .stFormSubmitButton > button:hover {{ background:var(--z-red-dark); border-color:var(--z-red-dark); color:white; }}
      .stDownloadButton > button {{ border-radius:12px; }}
      [data-testid="stMetric"] {{ background:#fff; border:1px solid var(--z-border); border-radius:16px; padding:14px 16px; }}
      [data-testid="stMetricLabel"] {{ color:#81767c; }}
      [data-testid="stMetricValue"] {{ color:var(--z-ink); }}
      .small-muted {{ color:var(--z-muted); font-size:.84rem; }}
      .prediction-placeholder {{ border:1px dashed #e8a7af; background:#fff8f9; border-radius:20px; padding:22px; }}
      .prediction-placeholder h3 {{ margin-top:0; color:var(--z-red-dark); }}
      div[data-testid="stDataFrame"] {{ border:1px solid var(--z-border); border-radius:14px; overflow:hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------- Helpers ------------------------------------
def pick_col(frame: pd.DataFrame, *candidates: str):
    """Return the first matching column name, ignoring case and outer spaces."""
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for candidate in candidates:
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_number(value, decimals=1, suffix=""):
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}{suffix}"


def add_metric_row(items):
    columns = st.columns(len(items))
    for col, (label, value, help_text) in zip(columns, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{value}</div>
                  <div class="small-muted" style="margin-top:7px">{help_text}</div>
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
        margin=dict(l=12, r=12, t=42, b=12),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="DM Sans, sans-serif", color=INK),
        title_font=dict(family="Manrope, sans-serif", size=16, color=INK),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#F4E6E8", zeroline=False)
    return fig


def mode_or_unknown(series):
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return "Not available"
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else "Not available"

# ------------------------------ Header ---------------------------------------
hero_left, hero_right = st.columns([8, 1.1], vertical_alignment="center")
with hero_left:
    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="hero-kicker">Delivery operations • Machine learning portfolio</div>
            <div class="hero-title">Zomato Delivery Intelligence</div>
            <p class="hero-subtitle">Explore delivery performance, discover operational clusters on the map, and prepare a delivery-time estimate from one clean workspace.</p>
          </div>
          <div class="pill">DATA • MAPS • PREDICTIONS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    try:
        st.image(LOGO_URL, width=82)
    except Exception:
        st.markdown('<div class="brand-fallback">zomato</div>', unsafe_allow_html=True)

# ------------------------- Load data / sidebar -------------------------------
st.sidebar.markdown(
    f"""
    <div style="padding:8px 0 14px 0">
      <div style="font-size:.75rem; letter-spacing:.14em; text-transform:uppercase; color:{ZOMATO_RED}; font-weight:800">Workspace</div>
      <div style="font-family:Manrope,sans-serif; font-size:1.25rem; font-weight:800; color:{INK}; margin-top:3px">Delivery Control</div>
      <div style="font-size:.85rem; color:{MUTED}; margin-top:5px">Insights at a glance</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader("Upload your cleaned Zomato CSV", type=["csv"])

def load_data():
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file), f"Uploaded: {uploaded_file.name}"
        except Exception as exc:
            st.sidebar.error(f"Could not read CSV: {exc}")
            return None, None

    likely_files = [
        "zomato_cleaned.csv",
        "zomato_delivery_cleaned.csv",
        "delivery_data.csv",
        "zomato.csv",
    ]
    for file_name in likely_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                return pd.read_csv(file_path), f"Loaded: {file_path.name}"
            except Exception as exc:
                st.sidebar.error(f"Could not read {file_path.name}: {exc}")
    return None, None


df, data_source = load_data()
if df is None:
    st.markdown(
        """
        <div class="soft-card">
          <h4>Bring your delivery dataset into the workspace</h4>
          <p>Place your cleaned CSV beside <code>app.py</code> using a name such as <code>zomato_cleaned.csv</code>, or upload it from the sidebar. The app expects your existing columns and does not train or modify ML models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Normalise header whitespace only (do not silently rename core columns).
df.columns = [str(c).strip() for c in df.columns]

# Detect columns from the dataset.
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
order_time_col = pick_col(df, "Time_Orderd", "Time_Ordered", "order_hour", "Time_Orderd_Hour")
restaurant_lat_col = pick_col(df, "Restaurant_latitude", "restaurant_latitude")
restaurant_lon_col = pick_col(df, "Restaurant_longitude", "restaurant_longitude")
delivery_lat_col = pick_col(df, "Delivery_location_latitude", "Delivery_latitude", "delivery_location_latitude")
delivery_lon_col = pick_col(df, "Delivery_location_longitude", "Delivery_longitude", "delivery_location_longitude")

# Coerce measures used in visuals to numeric without changing other source columns.
for column in [time_col, distance_col, rating_col, age_col, restaurant_lat_col, restaurant_lon_col, delivery_lat_col, delivery_lon_col]:
    if column is not None:
        df[column] = as_numeric(df[column])

st.sidebar.caption(data_source)
st.sidebar.markdown("---")
if city_col:
    city_options = sorted(df[city_col].dropna().astype(str).unique().tolist())
    chosen_cities = st.sidebar.multiselect("City filter", city_options, default=city_options)
    filtered_df = df[df[city_col].astype(str).isin(chosen_cities)].copy() if chosen_cities else df.iloc[0:0].copy()
else:
    filtered_df = df.copy()
    st.sidebar.caption("No City column detected; showing all records.")

st.sidebar.markdown("---")
st.sidebar.caption("UI skeleton only — prediction model is not connected yet.")

# Shared plot palette.
PALETTE = [ZOMATO_RED, "#FF7E8B", "#FFB1C0", "#B91F2B", "#4B3439", "#FFD9DE", "#AFA0A4"]

# ------------------------------- Main tabs -----------------------------------
tab_insights, tab_clusters, tab_prediction = st.tabs(
    ["📊  Delivery Insights", "🗺️  Clustering Intelligence", "⏱️  Delivery Predictor"]
)

# ================================= TAB 1 =====================================
with tab_insights:
    section_title("Delivery performance overview", "A visual snapshot of order volume, delivery duration, distance and operating conditions.")

    total_records = len(filtered_df)
    avg_time = filtered_df[time_col].mean() if time_col else np.nan
    avg_distance = filtered_df[distance_col].mean() if distance_col else np.nan
    avg_rating = filtered_df[rating_col].mean() if rating_col else np.nan

    add_metric_row([
        ("Total orders", f"{total_records:,}", "Records in current filter"),
        ("Avg. delivery time", format_number(avg_time, 1, " min"), "Mean of available delivery times"),
        ("Avg. distance", format_number(avg_distance, 2, " km"), "Uses your existing distance column" if distance_col else "Distance column not detected"),
        ("Avg. rider rating", format_number(avg_rating, 2, " / 5"), "Mean of available ratings"),
    ])

    st.write("")
    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        if time_col and filtered_df[time_col].notna().any():
            fig = px.histogram(
                filtered_df.dropna(subset=[time_col]), x=time_col, nbins=28,
                title="Delivery time distribution", color_discrete_sequence=[ZOMATO_RED],
                labels={time_col: "Delivery time (minutes)"}
            )
            fig.update_traces(marker_line_color="white", marker_line_width=1)
            st.plotly_chart(polish_plot(fig), use_container_width=True)
        else:
            st.info("Add a numeric `Time_taken (min)` column to view delivery-time distribution.")

    with chart_right:
        if traffic_col:
            traffic_data = filtered_df.copy()
            traffic_data[traffic_col] = traffic_data[traffic_col].astype("string").str.strip().str.title()
            if time_col:
                traffic_summary = traffic_data.groupby(traffic_col, dropna=False)[time_col].mean().reset_index()
                traffic_summary.columns = ["Traffic density", "Average delivery time (min)"]
                fig = px.bar(traffic_summary, x="Traffic density", y="Average delivery time (min)",
                             title="Average delivery time by traffic", color="Traffic density",
                             color_discrete_sequence=PALETTE, text_auto=".1f")
                st.plotly_chart(polish_plot(fig), use_container_width=True)
            else:
                traffic_summary = traffic_data[traffic_col].value_counts(dropna=False).reset_index()
                traffic_summary.columns = ["Traffic density", "Orders"]
                fig = px.bar(traffic_summary, x="Traffic density", y="Orders", title="Orders by traffic density",
                             color_discrete_sequence=[ZOMATO_RED], text_auto=True)
                st.plotly_chart(polish_plot(fig), use_container_width=True)
        else:
            st.info("No traffic-density column detected for this chart.")

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        if weather_col:
            weather_counts = filtered_df[weather_col].fillna("Missing").astype(str).str.strip().str.title().value_counts().reset_index()
            weather_counts.columns = ["Weather", "Orders"]
            fig = px.bar(weather_counts, x="Weather", y="Orders", title="Order volume by weather",
                         color="Weather", color_discrete_sequence=PALETTE, text_auto=True)
            st.plotly_chart(polish_plot(fig), use_container_width=True)
        else:
            st.info("No weather column detected for this chart.")

    with chart_right:
        if city_col and time_col:
            city_time = (
                filtered_df.dropna(subset=[city_col, time_col])
                .groupby(city_col)[time_col].mean().sort_values(ascending=False).head(12)
                .sort_values().reset_index()
            )
            city_time.columns = ["City", "Average delivery time (min)"]
            fig = px.bar(city_time, x="Average delivery time (min)", y="City", orientation="h",
                         title="Cities with the longest average delivery time",
                         color_discrete_sequence=[ZOMATO_RED], text_auto=".1f")
            st.plotly_chart(polish_plot(fig, height=390), use_container_width=True)
        elif distance_col and time_col:
            scatter_df = filtered_df.dropna(subset=[distance_col, time_col])
            if len(scatter_df) > 6000:
                scatter_df = scatter_df.sample(6000, random_state=42)
            fig = px.scatter(scatter_df, x=distance_col, y=time_col, title="Distance vs delivery time",
                             labels={distance_col: "Distance (km)", time_col: "Delivery time (min)"},
                             opacity=0.55, color_discrete_sequence=[ZOMATO_RED])
            st.plotly_chart(polish_plot(fig), use_container_width=True)
        else:
            st.info("Add a city or distance column to enable another performance view.")

    if distance_col and time_col:
        section_title("Distance and delivery time", "Use this relationship as an exploratory view; it does not imply causation.")
        scatter_df = filtered_df.dropna(subset=[distance_col, time_col])
        if len(scatter_df) > 7000:
            scatter_df = scatter_df.sample(7000, random_state=42)
        fig = px.scatter(
            scatter_df, x=distance_col, y=time_col,
            color=traffic_col if traffic_col else None,
            title="Delivery time against distance",
            labels={distance_col: "Distance (km)", time_col: "Delivery time (minutes)"},
            color_discrete_sequence=PALETTE, opacity=0.6,
            hover_data=[c for c in [city_col, weather_col, multiple_col] if c]
        )
        st.plotly_chart(polish_plot(fig, height=390), use_container_width=True)

# ================================= TAB 2 =====================================
with tab_clusters:
    section_title("Clustering intelligence", "Explore where orders are located and compare how delivery behavior differs by cluster.")

    if cluster_col is None:
        st.warning("No cluster-label column found. Expected `operational_cluster`, `geo_cluster`, or `cluster`.")
    else:
        cluster_values = filtered_df[cluster_col].dropna()
        cluster_labels = sorted(cluster_values.unique().tolist(), key=lambda x: str(x))
        non_noise_labels = [v for v in cluster_labels if str(v) != "-1"]
        n_clusters = len(non_noise_labels)
        noise_pct = (cluster_values.astype(str).eq("-1").mean() * 100) if len(cluster_values) else 0
        largest_cluster_size = int(filtered_df[cluster_col].value_counts().max()) if not filtered_df.empty else 0

        add_metric_row([
            ("Cluster count", f"{n_clusters:,}", f"Using `{cluster_col}` labels; -1 excluded"),
            ("Noise / unassigned", format_number(noise_pct, 1, "%"), "Rows labelled -1, if present"),
            ("Largest cluster", f"{largest_cluster_size:,}", "Records in the most populated cluster"),
            ("Mapped records", f"{len(filtered_df):,}", "Before map sampling"),
        ])
        st.write("")

        map_choices = []
        if delivery_lat_col and delivery_lon_col:
            map_choices.append("Delivery locations")
        if restaurant_lat_col and restaurant_lon_col:
            map_choices.append("Restaurant locations")

        map_left, map_right = st.columns([1.45, 1], gap="large")
        with map_left:
            section_title("Cluster map", "Map points are sampled to keep rendering responsive on larger datasets.")
            if not map_choices:
                st.warning("Could not find a complete latitude/longitude pair for restaurants or delivery locations.")
            else:
                map_type = st.radio("Show coordinates", map_choices, horizontal=True)
                if map_type == "Delivery locations":
                    lat_col, lon_col = delivery_lat_col, delivery_lon_col
                else:
                    lat_col, lon_col = restaurant_lat_col, restaurant_lon_col

                map_data = filtered_df.dropna(subset=[lat_col, lon_col, cluster_col]).copy()
                map_data = map_data[
                    map_data[lat_col].between(-90, 90)
                    & map_data[lon_col].between(-180, 180)
                ]
                if len(map_data) > 10000:
                    map_data = map_data.sample(10000, random_state=42)
                map_data["Cluster"] = map_data[cluster_col].astype(str)
                hover_cols = [c for c in [city_col, time_col, distance_col, traffic_col, weather_col] if c]
                if map_data.empty:
                    st.info("No valid coordinates are available under the current filters.")
                else:
                    fig = px.scatter_mapbox(
                        map_data,
                        lat=lat_col,
                        lon=lon_col,
                        color="Cluster",
                        hover_data=hover_cols,
                        zoom=3,
                        height=520,
                        mapbox_style="open-street-map",
                        title=f"{map_type} coloured by {cluster_col}",
                        color_discrete_sequence=PALETTE,
                    )
                    fig.update_traces(marker={"size": 7, "opacity": 0.72})
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=45, b=0),
                        paper_bgcolor="rgba(255,255,255,0)",
                        font=dict(family="DM Sans, sans-serif", color=INK),
                        legend_title_text="Cluster",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Displaying {len(map_data):,} valid points{ ' (sampled)' if len(filtered_df) > 10000 else '' }.")

        with map_right:
            section_title("Cluster explorer")
            cluster_options = sorted(filtered_df[cluster_col].dropna().unique().tolist(), key=lambda x: str(x))
            if cluster_options:
                selected_cluster = st.selectbox("Choose a cluster", cluster_options, format_func=lambda value: f"Cluster {value}")
                selected_data = filtered_df[filtered_df[cluster_col] == selected_cluster]
                selected_avg_time = selected_data[time_col].mean() if time_col else np.nan
                selected_avg_distance = selected_data[distance_col].mean() if distance_col else np.nan
                st.markdown(
                    f"""
                    <div class="soft-card">
                      <span class="pill">CLUSTER {selected_cluster}</span>
                      <h4 style="margin-top:14px">{len(selected_data):,} records</h4>
                      <p>Average delivery time: <b>{format_number(selected_avg_time, 1, ' min')}</b></p>
                      <p style="margin-top:7px">Average distance: <b>{format_number(selected_avg_distance, 2, ' km')}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if traffic_col:
                    traffic_mix = selected_data[traffic_col].fillna("Missing").astype(str).value_counts().reset_index()
                    traffic_mix.columns = ["Traffic", "Records"]
                    fig = px.pie(traffic_mix, names="Traffic", values="Records", hole=0.63,
                                 title="Traffic mix in selected cluster", color_discrete_sequence=PALETTE)
                    fig.update_layout(margin=dict(l=6, r=6, t=48, b=6), paper_bgcolor="rgba(255,255,255,0)")
                    st.plotly_chart(fig, use_container_width=True)
                if weather_col:
                    weather_mix = selected_data[weather_col].fillna("Missing").astype(str).value_counts().head(8).reset_index()
                    weather_mix.columns = ["Weather", "Records"]
                    st.markdown("**Weather composition**")
                    st.dataframe(weather_mix, use_container_width=True, hide_index=True)

        section_title("Cluster profile table", "Average metrics and dominant categories help give the numeric cluster labels an operational meaning.")
        aggregations = {"total_records": (cluster_col, "size")}
        if distance_col:
            aggregations["avg_distance_km"] = (distance_col, "mean")
        if time_col:
            aggregations["avg_delivery_time_min"] = (time_col, "mean")
        if rating_col:
            aggregations["avg_rider_rating"] = (rating_col, "mean")
        profile = filtered_df.groupby(cluster_col, dropna=False).agg(**aggregations).reset_index()
        for column, label in [
            (traffic_col, "dominant_traffic"),
            (weather_col, "dominant_weather"),
            (vehicle_condition_col, "dominant_vehicle_condition"),
            (multiple_col, "dominant_multiple_deliveries"),
            (city_col, "dominant_city"),
        ]:
            if column:
                dominant = filtered_df.groupby(cluster_col, dropna=False)[column].agg(mode_or_unknown).rename(label).reset_index()
                profile = profile.merge(dominant, on=cluster_col, how="left")
        numeric_cols = profile.select_dtypes(include="number").columns
        profile[numeric_cols] = profile[numeric_cols].round(2)
        st.dataframe(profile, use_container_width=True, hide_index=True)
        st.download_button(
            "Download cluster profile CSV",
            data=profile.to_csv(index=False).encode("utf-8"),
            file_name="zomato_cluster_profile.csv",
            mime="text/csv",
        )

# ================================= TAB 3 =====================================
with tab_prediction:
    section_title("Delivery-time predictor", "Enter the details of a delivery to prepare the inputs for your prediction model.")

    st.markdown(
        """
        <div class="soft-card" style="margin-bottom:18px">
          <h4>Prediction workspace</h4>
          <p>This tab contains the user interface only. The trained model and preprocessing pipeline are intentionally not loaded or called in this version.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cities = sorted(df[city_col].dropna().astype(str).unique().tolist()) if city_col else ["Select later"]
    weather_options = sorted(df[weather_col].dropna().astype(str).unique().tolist()) if weather_col else ["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]
    traffic_options = sorted(df[traffic_col].dropna().astype(str).unique().tolist()) if traffic_col else ["Low", "Medium", "High", "Jam"]
    order_options = sorted(df[order_type_col].dropna().astype(str).unique().tolist()) if order_type_col else ["Snack", "Meal", "Drinks", "Buffet"]
    vehicle_options = sorted(df[vehicle_type_col].dropna().astype(str).unique().tolist()) if vehicle_type_col else ["motorcycle", "scooter", "electric_scooter"]
    festival_options = sorted(df[festival_col].dropna().astype(str).unique().tolist()) if festival_col else ["No", "Yes"]

    with st.form("delivery_prediction_form", border=True):
        st.markdown("#### Order and location details")
        c1, c2, c3 = st.columns(3)
        with c1:
            input_city = st.selectbox("City", cities)
            input_order_type = st.selectbox("Order type", order_options)
            input_vehicle_type = st.selectbox("Vehicle type", vehicle_options)
        with c2:
            input_weather = st.selectbox("Weather conditions", weather_options)
            input_traffic = st.selectbox("Road traffic density", traffic_options)
            input_festival = st.selectbox("Festival", festival_options)
        with c3:
            input_vehicle_condition = st.selectbox("Vehicle condition", [0, 1, 2])
            input_multiple = st.selectbox("Multiple deliveries", [0, 1, 2, 3])
            input_order_hour = st.slider("Order hour", min_value=0, max_value=23, value=19)

        st.markdown("#### Delivery partner details")
        c1, c2 = st.columns(2)
        with c1:
            input_age = st.slider("Delivery person age", min_value=18, max_value=60, value=30)
        with c2:
            input_rating = st.slider("Delivery person rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)

        st.markdown("#### Coordinates (optional UI fields)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            input_rest_lat = st.number_input("Restaurant latitude", value=28.6139, format="%.6f")
        with c2:
            input_rest_lon = st.number_input("Restaurant longitude", value=77.2090, format="%.6f")
        with c3:
            input_delivery_lat = st.number_input("Delivery latitude", value=28.6200, format="%.6f")
        with c4:
            input_delivery_lon = st.number_input("Delivery longitude", value=77.2150, format="%.6f")

        submitted = st.form_submit_button("Prepare prediction inputs", use_container_width=True)

    if submitted:
        st.session_state["prediction_inputs"] = {
            "City": input_city,
            "Type_of_order": input_order_type,
            "Type_of_vehicle": input_vehicle_type,
            "Weather_conditions": input_weather,
            "Road_traffic_density": input_traffic,
            "Festival": input_festival,
            "Vehicle_condition": input_vehicle_condition,
            "multiple_deliveries": input_multiple,
            "Time_Orderd_Hour": input_order_hour,
            "Delivery_person_Age": input_age,
            "Delivery_person_Ratings": input_rating,
            "Restaurant_latitude": input_rest_lat,
            "Restaurant_longitude": input_rest_lon,
            "Delivery_location_latitude": input_delivery_lat,
            "Delivery_location_longitude": input_delivery_lon,
        }
        st.success("Inputs captured. No delivery-time prediction has been generated because the trained model is not connected in this UI skeleton.")

    placeholder_left, placeholder_right = st.columns([1.25, 1], gap="large")
    with placeholder_left:
        st.markdown(
            """
            <div class="prediction-placeholder">
              <span class="pill">MODEL OUTPUT PLACEHOLDER</span>
              <h3 style="margin-top:14px">Estimated delivery time</h3>
              <div style="font-family:Manrope,sans-serif;font-size:2.7rem;font-weight:800;color:#E23744;line-height:1.2">— min</div>
              <p style="margin-top:8px">Your saved preprocessing pipeline and regressor can be connected here later. This screen intentionally does not invent a prediction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with placeholder_right:
        section_title("Integration point")
        st.markdown(
            """
            <div class="soft-card">
              <h4>When your model is ready</h4>
              <p>Inside the submit handler, pass the captured form values through the same preprocessing used during training, call your saved regressor, and render the returned minutes in the output card.</p>
              <p style="margin-top:10px"><b>Keep training code outside this UI file.</b> Load your saved pipeline from a separate module or model file.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "prediction_inputs" in st.session_state:
            with st.expander("View captured form inputs"):
                st.json(st.session_state["prediction_inputs"])

# ------------------------------- Footer --------------------------------------
st.markdown(
    f"""
    <div style="border-top:1px solid {BORDER}; margin-top:34px; padding-top:16px; display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap; color:{MUTED}; font-size:.8rem">
      <span><b style="color:{ZOMATO_RED}">Zomato Delivery Intelligence</b> · Portfolio dashboard</span>
      <span>Exploration · Clustering · Prediction UI</span>
    </div>
    """,
    unsafe_allow_html=True,
)
