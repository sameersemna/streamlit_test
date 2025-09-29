import streamlit as st
import requests
import pandas as pd
# import io
# import sys
# sys.path.append("..")
# sys.path.append('../src')
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster, Draw
from datetime import datetime, date,timedelta

from constants import BACKEND_URL, DATA_TABLES, get_engine


# ----- new page -----
def run_user_interface():
    st.title("🌍 Berlin Interactive Map and Groundwater predictions")

    # ======= INSTRUCTION BOX =======

    st.markdown(
        """
        <div style="border:1px solid #ddd; padding:10px; border-radius:10px;background-color:#f9f9f9;box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">

        ### 👋 Welcome!

        Use this page to **explore groundwater stations in Berlin** and interact with them:

        - 🌍 **Map:** View official groundwater stations on the map (blue markers)
        - 🖱️ **Click:** Click on the map or drop markers to add your own custom points (purple stars)
        - ✍️ **Manual Entry:** Add a point by typing coordinates
        - 📅 **Date Filter:** Adjust the analysis period
        - 📥 **Export:** Download your added points as a CSV file
        </div>
        """,
        unsafe_allow_html=True
    )
    # User input section
    engine,_ = get_engine()


    # ---------- LOAD STATIONS ---------- via FastAPI, from DB
    if "stations_df" not in st.session_state:
        try:
            stations_df = pd.read_sql("SELECT * FROM stations_meta",engine)
            stations_df["ID"] = "Station " + stations_df["ID"].astype(str)
            st.session_state.stations_df = stations_df
            st.success(f"✅ Loaded {len(st.session_state.stations_df)} stations from Database")
        except Exception as e:
            st.error(f"❌ Error loading stations_meta: {e}")
            st.session_state.stations_df = pd.DataFrame()

    stations_df = st.session_state.stations_df


    # convert station ID to String
    if pd.api.types.is_numeric_dtype(stations_df["ID"]):
        stations_df["ID"] = "Station " + stations_df["ID"].astype(str)


    # ---------- LOAD STATIONS ---------- via Local computer
    # stations_df = pd.read_csv("./data/wasserportal/stations_groundwater.csv")
    # stations_df["ID"] = "Station " + stations_df["ID"].astype(str)


    # ---------- SESSION STATE ----------
    if "new_points" not in st.session_state:
        st.session_state.new_points = []
    if "rerun_flag" not in st.session_state:
        st.session_state.rerun_flag = 0
    if "slider_range" not in st.session_state:
        # st.session_state.slider_range = (stations_df["date"].min().date(), stations_df["date"].max().date())
        st.session_state.slider_range = (date(2025, 1, 1), date(2027, 1, 1))
    if "api_logs" not in st.session_state:
        st.session_state.api_logs = []
    if "current_run_logs" not in st.session_state:
        st.session_state.current_run_logs = []
    if "last_added_point" not in st.session_state:
            st.session_state.last_added_point = None

    # ---------- CREATE MAP FUNCTION ----------
    def create_map(stations, user_points):
        m = folium.Map(location=[52.52, 13.405], zoom_start=12, tiles="OpenStreetMap")

        # --- Stations layer with MarkerCluster ---
        stations_layer = folium.FeatureGroup(name="Stations", show=True)
        marker_cluster = MarkerCluster().add_to(stations_layer)
        for s in stations:
            popup_text = f"<b>{s['name']}</b><br>Lat: {s['lat']:.5f}<br>Lon: {s['lon']:.5f}"
            popup = folium.Popup(popup_text, max_width=300)
            marker=folium.Marker([s["lat"], s["lon"]], popup=popup, tooltip=s["name"],
                        icon=folium.Icon(color="blue", icon="info-sign"))
            # Add custom click handler
            marker.add_to(marker_cluster)
        stations_layer.add_to(m)

        # --- User-added points layer ---
        user_layer = folium.FeatureGroup(name="User Points", show=True)
        for p in user_points:
            popup_text = f"<b>{p['name']}</b><br>Lat: {p['lat']:.5f}<br>Lon: {p['lon']:.5f}"
            popup = folium.Popup(popup_text, max_width=300)
            folium.Marker([p["lat"], p["lon"]], popup=popup,
                        icon=folium.Icon(color="purple", icon="star")).add_to(user_layer)
        user_layer.add_to(m)

        folium.LayerControl().add_to(m)

        draw = Draw(
        draw_options={"polyline": False, "polygon": False, "circle": False, "rectangle": False, "circlemarker": False, "marker": True},
        edit_options={"edit": True}
        )
        m.add_child(draw)
        return m

    # ---------- FILTER STATIONS ----------
    # stations = stations_filtered.iloc[:100].rename(columns={"ID": "name"}).to_dict(orient="records")
    # stations = stations_df.iloc[::3].rename(columns={"ID": "name"}).to_dict(orient="records")
    stations = stations_df.iloc[:100].rename(columns={"ID": "name"}).to_dict(orient="records")

    # Map + Added points side by side
    st.markdown("### 🔎 Explore the map and track your custom points")
    # After loading stations
    with st.expander("📊 Station Table (from Database)"):
        st.markdown(
            "This table shows the **groundwater stations** retrieved from the FastAPI backend. "
            "Each station has an ID, name, and location (lat/lon)."
        )
        st.dataframe(stations_df)

    st.divider()

    # Create two columns with custom width ratios
    col_map, col_table = st.columns([2, 1])   # 2/3 width for map, 1/3 for table

    with col_map:
        st.subheader("🌍 Map of Stations")
        st.markdown("Blue markers = official stations, Purple = your custom points.")
        m = create_map(stations, st.session_state.new_points)
        map_data = st_folium(m, width=900, height=600, key=f"map_{st.session_state.rerun_flag}")

    with col_table:
        st.subheader("📋 Added Points")
        st.markdown("Points you clicked on or added manually will appear below.")
        df = pd.DataFrame(st.session_state.new_points)

        # Show success message here instead of directly after adding
        # if st.session_state.last_added_point:
        #     lat = st.session_state.last_added_point["lat"]
        #     lon = st.session_state.last_added_point["lon"]
        #     name = st.session_state.last_added_point["name"]
        #     st.success(f"Added {name} at {lat:.5f}, {lon:.5f}")
        if "last_added_message" in st.session_state:
            st.success(st.session_state.last_added_message)

        if not df.empty:
            st.dataframe(df, use_container_width=True, height=450)
            csv = df.to_csv(index=False).encode("utf-8")

            # Create two columns for the buttons
            btn_col1, btn_col2 = st.columns([1, 1])

            with btn_col1:
                st.download_button(
                    "📥 Download as CSV",
                    data=csv,
                    file_name="new_points.csv",
                    mime="text/csv",
                )

            with btn_col2:
                if st.button("🧹 Clear added points"):
                    st.session_state.new_points = []
                    st.session_state.rerun_flag += 1
                    st.success("Cleared all added points.")
                    st.session_state.last_added_message = None
        else:
            st.info("No points added yet. Try clicking on the map or using the form below!")


    # Controls
    st.divider()
    st.markdown("### ⚙️ Controls")

    # ---------- MAP DISPLAY ----------
    # m = create_map(stations, st.session_state.new_points)
    # map_data = st_folium(m, width=900, height=600, key=f"map_{st.session_state.rerun_flag}")

    # ---------- PANELS BELOW MAP ----------
    col_add, col_date, col_empty1, col_empty2 = st.columns(4)

    # --- Add Manual Point ---
    with col_add:
        st.markdown(
            '<h4 style="font-weight:600;">➕ Add a new point manually</h4>',
            unsafe_allow_html=True
            )
        with st.form("manual_point_form", clear_on_submit=True):
            name = st.text_input("Point name", f"New Point {len(st.session_state.new_points)+1}")
            lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=52.52, format="%.5f")
            lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=13.405, format="%.5f")
            submitted = st.form_submit_button("Add Point")

            if submitted:
                new_point = {"id": len(st.session_state.new_points)+1, "lat": lat, "lon": lon, "name": name}
                st.session_state.new_points.append(new_point)
                st.session_state.last_added_point = new_point
                st.session_state.last_added_message = f"Manually added  {new_point['name']} at {lat:.5f}, {lon:.5f}"
                st.session_state.rerun_flag += 1
                st.success(f"Manually added {name} at {lat:.5f}, {lon:.5f}")

    # --- Date Range Filter ---
    with col_date:

        st.markdown(
            '<h4 style="font-weight:600;">📅 Date Range Filter</h4>',
            unsafe_allow_html=True
            )
        slider_range = st.slider(
            "Select date range",
            min_value=date(2025, 1, 1),
            max_value=date(2027, 1, 1),
            value=st.session_state.slider_range,
            key="date_slider_widget"
        )
        start_input = st.text_input("Start date (YYYY-MM-DD)", value=str("2025-04-01"), key="start_input")
        end_input = st.text_input("End date (YYYY-MM-DD)", value=str("2025-04-30"), key="end_input")

        # Sync slider and text input
        try:
            start_dt = pd.to_datetime(start_input)
            end_dt = pd.to_datetime(end_input)
            if (start_dt.date(), end_dt.date()) != st.session_state.slider_range:
                st.session_state.slider_range = (start_dt.date(), end_dt.date())
                slider_range = st.session_state.slider_range
        except Exception:
            st.error("Invalid date format. Use YYYY-MM-DD.")

    # ---------- HANDLE MAP CLICKS ----------
    # if map_data and map_data.get("last_clicked"):
    #     lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
    #     matched_station = next((s for s in stations if abs(lat - s["lat"]) < 1e-2 and abs(lon - s["lon"]) < 1e-2), None)
    #     if matched_station:
    #         new_point = {"id": len(st.session_state.new_points)+1,
    #                     "lat": matched_station["lat"],
    #                     "lon": matched_station["lon"],
    #                     "name": f"New Point {len(st.session_state.new_points)+1} ({matched_station['name']})"}
    #     else:
    #         new_point = {"id": len(st.session_state.new_points)+1,
    #                     "lat": lat, "lon": lon,
    #                     "name": f"New Point {len(st.session_state.new_points)+1}"}
    #     st.session_state.new_points.append(new_point)
    #     st.session_state.rerun_flag += 1
    #     st.session_state.last_added_point = new_point
    #     st.success(f"Added {new_point['name']} at {lat:.5f}, {lon:.5f}")

    # ---------- HANDLE MAP CLICKS ----------
    if map_data:
        # Case 1 — Clicked on a map object (marker)
        if map_data.get("last_object_clicked"):
            obj = map_data["last_object_clicked"]

            if "tooltip" in obj:
                station_name = obj["tooltip"]
                matched_station = next((s for s in stations if s["name"] == station_name), None)

                if matched_station:
                    new_point = {
                        "id": len(st.session_state.new_points) + 1,
                        "lat": matched_station["lat"],
                        "lon": matched_station["lon"],
                        "name": matched_station["name"]
                    }
                    st.session_state.new_points.append(new_point)
                    st.session_state.rerun_flag += 1
                    st.session_state.last_added_message = f"Added {matched_station['name']} at {matched_station['lat']:.5f}, {matched_station['lon']:.5f}"

        # Case 2 — Clicked anywhere on the map
        elif map_data.get("last_clicked"):
            lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
            matched_station = next((s for s in stations if abs(lat - s["lat"]) < 1e-2 and abs(lon - s["lon"]) < 1e-2), None)
            if matched_station:
                new_point = {"id": len(st.session_state.new_points)+1,
                            "lat": matched_station["lat"],
                            "lon": matched_station["lon"],
                            "name": f"New Point {len(st.session_state.new_points)+1} ({matched_station['name']})"}
            else:
                new_point = {"id": len(st.session_state.new_points)+1,
                            "lat": lat, "lon": lon,
                            "name": f"New Point {len(st.session_state.new_points)+1}"}
            st.session_state.new_points.append(new_point)
            st.session_state.rerun_flag += 1
            st.session_state.last_added_point = new_point
            st.session_state.last_added_message = f"Added {new_point['name']} at {lat:.5f}, {lon:.5f}"
            st.success(f"Added {new_point['name']} at {lat:.5f}, {lon:.5f}")

        # Case 3 — Clicked on a drawing, drop a point via DARW tool
        elif "last_active_drawing" in map_data:
            drawing = map_data["last_active_drawing"]
            if drawing and drawing["geometry"]["type"] == "Point":
                lat = drawing["geometry"]["coordinates"][1]
                lon = drawing["geometry"]["coordinates"][0]

                matched_station = next((s for s in stations if abs(lat - s["lat"]) < 1e-2 and abs(lon - s["lon"]) < 1e-2), None)
                if matched_station:
                    new_point = {"id": len(st.session_state.new_points)+1,
                                "lat": matched_station["lat"],
                                "lon": matched_station["lon"],
                                "name": f"New Point {len(st.session_state.new_points)+1} ({matched_station['name']})"}
                else:
                    new_point = {"id": len(st.session_state.new_points)+1,
                                "lat": lat, "lon": lon,
                                "name": f"New Point {len(st.session_state.new_points)+1}"}
                st.session_state.new_points.append(new_point)
                st.session_state.rerun_flag += 1
                st.session_state.last_added_point = new_point
                st.session_state.last_added_message = f"Drawed  {new_point['name']} at {lat:.5f}, {lon:.5f}"
                st.success(f"Added {new_point['name']} at {lat:.5f}, {lon:.5f}")

    # -------------------
    # SECTION: Send Data to API for Predictions
    # -------------------

    if st.session_state.new_points and st.session_state.slider_range:

        st.markdown(
            """
            <div style="
                border:1px solid #ddd;
                padding:15px;
                border-radius:10px;
                background-color:#f9f9f9;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">
            <h4 style="font-weight:600;">📈 Get Predictions & History</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Fetch Predictions from database"):
            with st.spinner("Sending request to database..."):
                start_date = st.session_state.slider_range[0]
                end_date = st.session_state.slider_range[1]
                points = pd.DataFrame(st.session_state.new_points)

                # for prediction only seven days increment
                pred_date = start_date + timedelta(days=7)
                start_date_bevor30 = start_date - timedelta(days=120)



                st.write(f"Start Date: {start_date}")
                st.write(f"End Date: {end_date}")
                # st.dataframe(points)

                st.write(f"Display stations that are closest to User selected points and their groundwater level history")

                # Convert datetime objects to strings
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
                pred_date_str = pred_date.strftime("%Y-%m-%d")

                # find closest stations to the new points
                user_lats=points['lat']
                user_lons=points['lon']

                all_stations = st.session_state.stations_df

                all_stations["ID"] = all_stations["ID"].str.split().str[-1].astype(int)

                nearby_stations = []
                obs_df_list = []
                pred_df_list = []
                i=0
                for lat, lon in points[["lat","lon"]].values:
                    dist = (all_stations["lat"] - lat)**2 + (all_stations["lon"] - lon)**2
                    min_idx = dist.idxmin()
                    closest_station = all_stations.loc[min_idx]
                    nearby_stations.append(closest_station)
                    # closest_station['ID'] = 100
                    points.loc[i, 'Closest Station ID'] = closest_station['ID']
                    i+=1

                    gw_cs_df = pd.read_sql(f"SELECT * FROM gw_table WHERE station = {int(closest_station['ID'])} AND date BETWEEN '{start_date_bevor30}' AND '{start_date_str}'", engine)
                    cs_pred_df = pd.read_sql(f"SELECT * FROM gw_table WHERE station = {int(closest_station['ID'])} AND date BETWEEN '{start_date_str}' AND '{pred_date_str}'", engine)

                    # st.subheader("Groundwater Level History:")
                    # st.dataframe(gw_cs_df.set_index("date"))

                    # if not cs_pred_df.empty:
                    #     st.subheader("Predicted Groundwater Level (7 days):")
                    #     st.dataframe(cs_pred_df.set_index("date"))

                    obs_df_list.append(gw_cs_df)
                    pred_df_list.append(cs_pred_df)

                st.dataframe(points)
                gw_cs_df_all = pd.concat(obs_df_list)
                cs_pred_df_all = pd.concat(pred_df_list)

                st.info("To be able to plot multiple stations in a single figure, the station mean groundwater level is removed respectively.")
                # make plotly figures
                # fig = create_obs_pred_fig(gw_cs_df, cs_pred_df)
                fig = plot_multiple_stations(gw_cs_df_all, cs_pred_df_all)
                st.plotly_chart(fig, use_container_width=True)


    else:
        st.info("Select at least one point and a date range to fetch predictions.")



import plotly.graph_objs as go

def create_obs_pred_fig(obs_df, pred_df, y_range=None, title="Observations vs Predictions"):
    """
    Plot observations and predictions in different colors in Streamlit.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Must have 'timestamp' and 'value' columns for observations.
    pred_df : pd.DataFrame
        Must have 'timestamp' and 'value' columns for predictions.
    y_range : tuple (min, max), optional
        Y-axis range to fix. If None, auto-scale.
    title : str
        Chart title.
    """
    fig = go.Figure()

    # Observations trace (past)
    fig.add_trace(
        go.Scatter(
            x=obs_df["date"],
            y=obs_df["value"],
            mode="lines",
            name="Observations",
            line=dict(color="blue"),
        )
    )

    # Predictions trace (future)
    fig.add_trace(
        go.Scatter(
            x=pred_df["date"],
            y=pred_df["value"],
            mode="lines",
            name="Predictions",
            line=dict(color="red", dash="dash"),  # dashed for distinction
        )
    )

    if y_range:
        fig.update_yaxes(range=y_range)

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            rangeslider=dict(visible=True, thickness=0.15),
            type="date",
        ),
        title=title,
    )

    return fig


def plot_multiple_stations(obs_df, pred_df, y_range=None, title="Observations vs Predictions", height=600):
    """
    Plot multiple stations with observations and predictions in different colors.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Must have 'station_id', 'timestamp', 'value'.
    pred_df : pd.DataFrame
        Must have 'station_id', 'timestamp', 'value'.
    y_range : tuple (min, max), optional
        Y-axis range to fix. If None, auto-scale.
    title : str
        Chart title.
    """
    fig = go.Figure()

    stations = obs_df["station"].unique()

    for station in stations:
        obs_station = obs_df[obs_df["station"] == station]
        pred_station = pred_df[pred_df["station"] == station]

        obs_mean = obs_station["value"].mean()
        obs_station["value"] = obs_station["value"] - obs_mean
        pred_station["value"] = pred_station["value"] - obs_mean

        # Observations
        fig.add_trace(
            go.Scatter(
                x=obs_station["date"],
                y=obs_station["value"],
                mode="lines",
                name=f"{station} - Obs",
                line=dict(color=None),  # Let Plotly auto-choose
                legendgroup=f"{station}",  # 🔗 Link both traces in the legend
            )
        )

        # Predictions
        fig.add_trace(
            go.Scatter(
                x=pred_station["date"],
                y=pred_station["value"],
                mode="lines",
                name=f"{station} - Pred",
                line=dict(dash="dash", color=None),  # Auto color
                legendgroup=f"{station}",  # 🔗 Link both traces in the legend
                showlegend=False,
            )
        )

    if y_range:
        fig.update_yaxes(range=y_range)

    fig.update_layout(
        height=height,
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            ),
            rangeslider=dict(visible=True, thickness=0.15),
            type="date",
            tickformat="%Y-%m-%d",  # ✅ date format yyyy-mm-dd
            showgrid=True,           # ✅ x-axis grid lines
            gridcolor="LightGray",   # grid color
            gridwidth=1,             # grid line width
        ),
        yaxis=dict(
            showgrid=True,           # ✅ y-axis grid lines
            gridcolor="LightGray",
            gridwidth=1,

        ),
        title=title,
    )
    return fig
