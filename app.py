import streamlit as st
import requests
import pandas as pd

from constants import BACKEND_URL, DATA_TABLES, SQL_TABLE_LIMIT_DEFAULT, TITLE, BACKEND_PORT, get_engine
from pathlib import Path
# import io
# import sys
# sys.path.append("..")
# sys.path.append('../src')
import logging
import threading
import time
from prometheus_client import start_http_server, Counter, Gauge, CollectorRegistry

# --- 1. Constants and Initialization Guards ---
METRICS_PORT = 8502

# Define a key for the session state to track initialization
INIT_KEY = 'metrics_initialized'
# Get the default Prometheus registry
REGISTRY = CollectorRegistry() 


# --- 2. Custom Log Handler for Metrics ---
class PrometheusLogHandler(logging.Handler):
    """A log handler that increments the APP_LOG_ERRORS_TOTAL metric on errors."""
    def __init__(self, error_counter):
        super().__init__()
        self.error_counter = error_counter
        
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            self.error_counter.inc()


# --- 3. Initialization Logic (Runs ONLY ONCE) ---
# This block uses st.session_state to ensure that metrics, logging, 
# and the metrics server are set up only on the first run.
if INIT_KEY not in st.session_state:
    
    # Define custom metrics using the explicit registry
    # This metric counts how many times the Streamlit button is clicked
    BUTTON_CLICKS_TOTAL = Counter(
        'streamlit_button_clicks_total', 
        'Total number of times the primary button was clicked',
        registry=REGISTRY # Use the specific registry
    )
    # This metric counts every time an ERROR level log is generated
    APP_LOG_ERRORS_TOTAL = Counter(
        'streamlit_app_log_errors_total', 
        'Total count of application log errors generated',
        registry=REGISTRY # Use the specific registry
    )

    # Configure logger to use our custom handler
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Attach the custom handler using the defined metric
    logger.addHandler(PrometheusLogHandler(APP_LOG_ERRORS_TOTAL))
    
    # Start the metrics server
    def start_metrics_server():
        """Starts the Prometheus HTTP server in a non-blocking thread."""
        try:
            # Serve the metrics from our specific registry
            start_http_server(METRICS_PORT, registry=REGISTRY) 
            print(f"Prometheus metrics server started on port {METRICS_PORT}")
        except Exception as e:
            # Handle the case if the port is already in use
            print(f"Error starting metrics server (likely port in use): {e}")

    # Start the server thread
    thread = threading.Thread(target=start_metrics_server)
    thread.daemon = True
    thread.start()

    # Store necessary objects in session state for access during script reruns
    st.session_state[INIT_KEY] = {
        'button_counter': BUTTON_CLICKS_TOTAL,
        'logger': logger
    }

# --- 4. Streamlit UI Logic (Uses initialized objects from session state) ---

# Retrieve initialized objects from session state
BUTTON_CLICKS_TOTAL = st.session_state[INIT_KEY]['button_counter']
logger = st.session_state[INIT_KEY]['logger']

engine, _ = get_engine()

st.set_page_config(
    page_title=TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Streamlit specific code ----
st.title(TITLE)
st.sidebar.title("Table of contents")
pages = [
    "Intro", "Data", "Vizualization", "Modelling", "Train & Predict"
]
page = st.sidebar.radio("Go to", pages)

# --- Page 1 ---
if page == pages[0]:

    st.subheader("Project Objective")
    st.markdown(
        """
The goal of this project is to develop a system capable of analyzing weather and groundwater time series data in Berlin in order to detect potential basement flooding risks. Based on this analysis, users will receive early alerts, allowing them to take preventive actions (e.g., removing valuables or equipment from basements).

**To achieve this, two main datasets have been used:**

- **Daily precipitation levels on a 1 km grid** from the DWD Precipitation Data portal. This dataset provides daily rainfall measurements at a high spatial resolution (1 km²) across the Berlin area. These data help identify spatial and temporal rainfall patterns that may be linked to extreme weather events.

- **Groundwater measurements at 892 stations** from the Berlin Groundwater Measurements portal. This dataset contains daily groundwater level recordings from 892 stations throughout Berlin. These values are essential for assessing how groundwater levels respond to rainfall events, which can help model potential soil saturation and flooding conditions.
        """
    )

    
    st.divider()
    st.subheader("Project Architecture")  

    img_path = Path(__file__).parent / "assets" / "architecture.jpg"
    st.image(str(img_path), caption="High-level system architecture", use_column_width=True)


    st.markdown(f"Streamlit Metrics available for scraping by Prometheus at `http://localhost:{METRICS_PORT}/metrics`")
    st.markdown(f"FastAPI Metrics available for scraping by Prometheus at `http://localhost:{BACKEND_PORT}/metrics`")


# --- Page 2 ---
if page == pages[1]:
    st.header("Data")

    from pathlib import Path
    assets_dir = Path(__file__).parent / "assets"
    img_options = {
        "DWD grid (Berlin)": assets_dir / "dwd_grid_berlin.png",
        "Groundwater stations": assets_dir / "groundwater_station_locations.png",
    }

    st.subheader("Maps")
    choice = st.radio(
        label="",  
        options=list(img_options.keys()),
        horizontal=True,
        label_visibility="collapsed", 
    )
    st.image(str(img_options[choice]), caption=choice, use_column_width=True)


    
    st.title("Current Tables from Database")

    selected_table = st.selectbox(
        "Choose a Table:",
        DATA_TABLES,
        format_func=lambda x: f"{DATA_TABLES.get(x)} ({x})")
    selected_table_limit = st.number_input("Enter number of rows",
                                           value=SQL_TABLE_LIMIT_DEFAULT)

    if selected_table and st.button("Get Table"):
        try:
            with st.spinner('Fetching table...'):
                response = requests.get(f"{BACKEND_URL}/show_table", {
                    "table": selected_table,
                    "limit": selected_table_limit
                })

            if response.status_code == 200:
                table_data = response.json()
                if table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df)
                else:
                    st.info(
                        "No data found in the database. Upload a CSV file above."
                    )
            else:
                st.error(
                    f"Error fetching data. Status code: {response.status_code}"
                )
                st.json(response.json())

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the backend: {e}")
    st.divider()

    st.title("Update Database from APIs")
    if st.button("Update Data"):
        try:
            with st.spinner('Fetching API data...'):
                response = requests.get(f"{BACKEND_URL}/update_db")

            json_response = response.json()
            st.json(json_response)
            
            if json_response['fetched_wasserportal'] is not None:
                fetched_wasserportal = pd.read_parquet(f"{json_response['fetched_wasserportal']['file']}")
                st.markdown(f"\n**{json_response['fetched_wasserportal']['file']}**\n")
                st.dataframe(fetched_wasserportal.head())
                
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the backend: {e}")

# --- New Page: Train & Predict ---
if page == "Train & Predict":
    st.header("Train & Predict Groundwater Levels")
    subquery = "SELECT UNIQUE(station) FROM gw_table ORDER BY station"
    df_stations = pd.read_sql(f"SELECT id, lat, lon, height, id AS station_id FROM stations_meta WHERE id IN ({subquery}) ORDER BY id", engine)
    df_stations = df_stations.set_index('station_id')

    # st.subheader(df_stations.iloc[1]['lat'])
    # st.subheader(df_stations.iloc[2]['lon'])
    # st.subheader(df_stations.iloc[3]['height'])
    st.subheader("Train Model")
    with st.form("train_form"):
        # station_ids = st.text_input("Station IDs (comma separated)",
        #                             value="100")
        station_id_select = st.multiselect(
            "Choose a Station:",
            df_stations,
            max_selections=1,
            format_func=lambda x: (
                f"Station No. {x} "
                f"({df_stations.loc[int(x), 'lat']}, "
                f"{df_stations.loc[int(x), 'lon']}) "
                f"[Ht: {df_stations.loc[int(x), 'height']}]"
            )
        )
        start_date = st.text_input("Start Date (YYYY-MM-DD)",
                                   value="2022-01-01")
        end_date = st.text_input("End Date (YYYY-MM-DD)", value="2025-04-30")
        test_size = st.number_input("Test Size (0 for no test split)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.2)
        submitted_train = st.form_submit_button("Train Model")

        if station_id_select and submitted_train:
            try:
                # station_ids = [str(item) for item in station_id_select]
                # station_ids = ",".join(station_ids)
                # ids = [int(s.strip()) for s in station_ids.split(",") if s.strip()]
                ids = station_ids = [str(item) for item in station_id_select]
                payload = {
                    "station_ids": ids,
                    "start_date": start_date,
                    "end_date": end_date,
                    "test_size": test_size
                }
                with st.spinner("Training model..."):
                    response = requests.post(f"{BACKEND_URL}/train", json=payload)
                st.write("Response:")
                st.json(response.json())
            except Exception as e:
                st.error(f"Training request failed: {e}")

    st.divider()

    st.subheader("Predict Groundwater Levels")
    with st.form("predict_form"):
        # station_id_pred = st.number_input("Station ID", min_value=0, value=100)
        station_id_pred = st.selectbox(
            "Choose a Station:",
            df_stations,            
            format_func=lambda x: (
                f"Station No. {x} "
                f"({df_stations.loc[int(x), 'lat']}, "
                f"{df_stations.loc[int(x), 'lon']}) "
                f"[Ht: {df_stations.loc[int(x), 'height']}]"
            )
        )
        start_date_pred = st.text_input("Start Date (YYYY-MM-DD)",
                                        value="2025-01-01",
                                        key="pred_start")
        end_date_pred = st.text_input("End Date (YYYY-MM-DD)",
                                      value="2025-04-30",
                                      key="pred_end")
        submitted_predict = st.form_submit_button("Predict")

    if submitted_predict:
        try:
            params = {
                "station_id": station_id_pred,
                "start_date": start_date_pred,
                "end_date": end_date_pred
            }
            with st.spinner("Predicting..."):
                response = requests.get(f"{BACKEND_URL}/predict",
                                        params=params)
            st.write("Prediction run complete.")
            # st.json(response.json())
        except Exception as e:
            st.error(f"Prediction request failed: {e}")
