"""
Simplified Sales Forecasting Inference UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.simple_model_loader import (
    MODEL_ARTIFACTS,
    SimpleModelLoader,
    display_model_type,
)
from utils.simple_predictor import (
    FUTURE_REQUIRED_COLUMNS,
    HISTORICAL_REQUIRED_COLUMNS,
    SimplePredictor,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATE_HOLIDAY_OPTIONS = ["none", "a", "b", "c"]
STORE_TYPE_OPTIONS = ["a", "b", "c", "d"]
ASSORTMENT_OPTIONS = ["a", "b", "c"]
PROMO_INTERVAL_OPTIONS = [
    "none",
    "Jan,Apr,Jul,Oct",
    "Feb,May,Aug,Nov",
    "Mar,Jun,Sept,Dec",
]
BINARY_LABEL_TO_VALUE = {"No": 0, "Yes": 1}
VALUE_TO_BINARY_LABEL = {0: "No", 1: "Yes"}
BINARY_COLUMNS = [
    "has_promotion",
    "is_open",
    "school_holiday",
    "promo2",
]


def binary_select(label, value=0, key=None):
    return BINARY_LABEL_TO_VALUE[
        st.selectbox(
            label,
            list(BINARY_LABEL_TO_VALUE.keys()),
            index=int(value),
            key=key,
        )
    ]


def business_column_config(include_sales=True, disable_date=False):
    config = {
        "date": st.column_config.DateColumn("Date", disabled=disable_date),
        "store_id": st.column_config.TextColumn("Store ID", required=True),
        "customer_traffic": st.column_config.NumberColumn(
            "Customer Traffic",
            min_value=0,
            step=1,
            required=True,
        ),
        "has_promotion": st.column_config.SelectboxColumn(
            "Promotion",
            options=["No", "Yes"],
            required=True,
        ),
        "is_open": st.column_config.SelectboxColumn(
            "Open",
            options=["No", "Yes"],
            required=True,
        ),
        "school_holiday": st.column_config.SelectboxColumn(
            "School Holiday",
            options=["No", "Yes"],
            required=True,
        ),
        "state_holiday": st.column_config.SelectboxColumn(
            "State Holiday",
            options=STATE_HOLIDAY_OPTIONS,
            required=True,
        ),
        "store_type": st.column_config.SelectboxColumn(
            "Store Type",
            options=STORE_TYPE_OPTIONS,
            required=True,
        ),
        "assortment": st.column_config.SelectboxColumn(
            "Assortment",
            options=ASSORTMENT_OPTIONS,
            required=True,
        ),
        "competition_distance": st.column_config.NumberColumn(
            "Competition Distance",
            min_value=0,
            step=100,
            required=True,
        ),
        "promo2": st.column_config.SelectboxColumn(
            "Promo2",
            options=["No", "Yes"],
            required=True,
        ),
        "promo_interval": st.column_config.SelectboxColumn(
            "Promo Interval",
            options=PROMO_INTERVAL_OPTIONS,
            required=True,
        ),
    }
    if include_sales:
        config["sales"] = st.column_config.NumberColumn(
            "Sales",
            min_value=0,
            step=100,
            required=True,
        )
    return config


def display_binary_labels(df):
    df = df.copy()
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(int)
                .map(VALUE_TO_BINARY_LABEL)
                .fillna("No")
            )
    return df


def normalize_editor_data(df):
    df = df.copy()
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_LABEL_TO_VALUE).fillna(df[col]).astype(int)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "state_holiday" in df.columns and "school_holiday" in df.columns:
        normalized_state_holiday = (
            df["state_holiday"]
            .fillna("none")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        no_holiday_values = {"", "0", "0.0", "none", "nan", "nat", "false"}
        df["is_holiday"] = (
            (~normalized_state_holiday.isin(no_holiday_values))
            | (df["school_holiday"] == 1)
        ).astype(int)
    return df


def validate_input_columns(df, required_cols):
    return [col for col in required_cols if col not in df.columns]


def validate_single_store(df):
    if "store_id" not in df.columns:
        return False
    return df["store_id"].astype(str).nunique() == 1


def single_store_error():
    st.error(
        "The UI currently supports one store at a time. "
        "Please upload data for a single store."
    )


def latest_value(input_data, column, fallback):
    if input_data is not None and column in input_data.columns and len(input_data) > 0:
        value = input_data[column].iloc[-1]
        if pd.notna(value):
            return value
    return fallback


def future_feature_template(input_data, forecast_days):
    last_date = pd.to_datetime(input_data["date"]).max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D",
    )

    rows = []
    for date_value in future_dates:
        rows.append(
            {
                "date": date_value,
                "store_id": latest_value(input_data, "store_id", "store_0001"),
                "customer_traffic": float(
                    latest_value(input_data, "customer_traffic", 0)
                ),
                "has_promotion": int(latest_value(input_data, "has_promotion", 0)),
                "is_open": int(latest_value(input_data, "is_open", 1)),
                "school_holiday": int(latest_value(input_data, "school_holiday", 0)),
                "state_holiday": latest_value(input_data, "state_holiday", "none"),
                "store_type": latest_value(input_data, "store_type", "a"),
                "assortment": latest_value(input_data, "assortment", "a"),
                "competition_distance": float(
                    latest_value(input_data, "competition_distance", 0)
                ),
                "promo2": int(latest_value(input_data, "promo2", 0)),
                "promo_interval": latest_value(input_data, "promo_interval", "none"),
            }
        )
    return pd.DataFrame(rows)


def base_manual_rows(days=7):
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": ["store_0001"] * days,
            "sales": [5000 + i * 100 for i in range(days)],
            "customer_traffic": [500] * days,
            "has_promotion": [0] * days,
            "is_open": [1] * days,
            "school_holiday": [0] * days,
            "state_holiday": ["none"] * days,
            "store_type": ["a"] * days,
            "assortment": ["a"] * days,
            "competition_distance": [500.0] * days,
            "promo2": [0] * days,
            "promo_interval": ["none"] * days,
        }
    )

# Page config
st.set_page_config(
    page_title="Sales Forecast Inference",
    layout="wide"
)

# Initialize session state
if 'model_loader' not in st.session_state:
    st.session_state.model_loader = SimpleModelLoader()
    st.session_state.predictor = SimplePredictor(st.session_state.model_loader)
    st.session_state.models_loaded = False
    st.session_state.run_id = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# Header
st.title("Sales Forecast Inference")
st.markdown("Generate sales predictions using trained ML models")

# Sidebar for model loading
with st.sidebar:
    st.header("Model Configuration")
    
    if not st.session_state.models_loaded:
        st.warning("No models loaded")
    else:
        st.success("Models loaded")
        loaded_models = [
            display_model_type(model_type)
            for model_type in st.session_state.model_loader.models.keys()
        ]
        st.info(f"Models: {', '.join(loaded_models)}")
        if st.session_state.run_id:
            st.caption(f"Run ID: {st.session_state.run_id[:8]}...")
    
    if st.button("Load/Reload Models", type="primary", use_container_width=True):
        with st.spinner("Loading models..."):
            run_id = st.session_state.model_loader.get_latest_run()
            if not run_id:
                st.error("No trained models found in MLflow. Please train a model first.")
            
            if run_id and st.session_state.model_loader.load_models_from_run(run_id):
                st.session_state.models_loaded = True
                st.session_state.run_id = run_id
                st.success("Models loaded!")
                st.rerun()
            else:
                st.error("Failed to load models")
    
    st.markdown("---")
    
    # Model selection
    model_options = (
        st.session_state.model_loader.available_model_types()
        if st.session_state.models_loaded
        else list(MODEL_ARTIFACTS.keys())
    )
    model_type = st.selectbox(
        "Model Type",
        model_options,
        format_func=display_model_type,
        help="The calibrated ensemble uses the saved training-time blend"
    )
    
    # Forecast settings
    forecast_days = st.slider(
        "Forecast Days",
        min_value=1,
        max_value=90,
        value=30
    )

# Main content
if st.session_state.models_loaded:
    # Input tabs
    tab1, tab2, tab3 = st.tabs(["Upload Data", "Manual Entry", "Sample Data"])
    
    input_data = None
    
    with tab1:
        st.markdown("### Upload Historical Sales Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV must include date, sales, store, traffic, promotion, holiday, and store metadata fields"
        )
        
        if uploaded_file is not None:
            input_data = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(input_data)} records")
            
            # Show preview
            with st.expander("Data Preview"):
                st.dataframe(input_data.head())
                
            # Basic validation
            missing_cols = validate_input_columns(
                input_data,
                HISTORICAL_REQUIRED_COLUMNS,
            )
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                input_data = None
            elif not validate_single_store(input_data):
                single_store_error()
                input_data = None
            else:
                input_data = normalize_editor_data(input_data)
                input_data = input_data.sort_values("date").reset_index(drop=True)
                st.session_state.input_data = input_data
    
    with tab2:
        st.markdown("### Enter Recent Sales Data")

        if "manual_editor_data" not in st.session_state:
            st.session_state.manual_editor_data = display_binary_labels(
                base_manual_rows(days=7)
            )

        edited_manual_data = st.data_editor(
            st.session_state.manual_editor_data,
            column_config=business_column_config(include_sales=True),
            num_rows="fixed",
            use_container_width=True,
            key="manual_feature_editor",
        )
        
        if st.button("Use Manual Data", key="manual_btn"):
            input_data = normalize_editor_data(edited_manual_data)
            if not validate_single_store(input_data):
                single_store_error()
                input_data = None
            else:
                input_data = input_data.sort_values("date").reset_index(drop=True)
                st.session_state.input_data = input_data
                st.session_state.manual_editor_data = display_binary_labels(input_data)
                st.success("Manual data ready for prediction")
    
    with tab3:
        st.markdown("### Generate Sample Data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_days = st.number_input("Historical Days", value=60, min_value=7)
        with col2:
            avg_sales = st.number_input("Average Daily Sales", value=5000, min_value=100)
        with col3:
            volatility = st.slider("Volatility (%)", 0, 50, 20)

        st.markdown("#### Sample Business Inputs")
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_store_id = st.text_input("Sample Store ID", value="store_0001")
            sample_store_type = st.selectbox("Store Type", STORE_TYPE_OPTIONS, key="sample_store_type")
            sample_assortment = st.selectbox("Assortment", ASSORTMENT_OPTIONS, key="sample_assortment")
            sample_competition_distance = st.number_input(
                "Competition Distance",
                min_value=0,
                value=500,
                step=100,
            )
        with col2:
            sample_customer_traffic = st.number_input(
                "Average Customer Traffic",
                min_value=0,
                value=500,
                step=50,
            )
            sample_has_promotion = binary_select(
                "Promotion",
                key="sample_has_promotion",
            )
            sample_is_open = binary_select(
                "Open",
                value=1,
                key="sample_is_open",
            )
            sample_promo2 = binary_select("Promo2", key="sample_promo2")
        with col3:
            sample_school_holiday = binary_select(
                "School Holiday",
                key="sample_school_holiday",
            )
            sample_state_holiday = st.selectbox(
                "State Holiday",
                STATE_HOLIDAY_OPTIONS,
                key="sample_state_holiday",
            )
            sample_promo_interval = st.selectbox(
                "Promo Interval",
                PROMO_INTERVAL_OPTIONS,
                key="sample_promo_interval",
            )
        
        if st.button("Generate Sample Data", key="sample_btn"):
            # Generate realistic sample data
            dates = pd.date_range(end=datetime.now(), periods=sample_days, freq='D')
            
            # Add trend and seasonality
            trend = np.linspace(0, avg_sales * 0.1, sample_days)
            seasonal = avg_sales * 0.2 * np.sin(2 * np.pi * np.arange(sample_days) / 7)
            noise = np.random.normal(0, avg_sales * volatility / 100, sample_days)
            
            sales = avg_sales + trend + seasonal + noise
            sales = np.maximum(sales, 0)  # Ensure non-negative
            
            input_data = pd.DataFrame({
                'date': dates,
                'store_id': sample_store_id,
                'sales': sales,
                'customer_traffic': sample_customer_traffic,
                'has_promotion': sample_has_promotion,
                'is_open': sample_is_open,
                'school_holiday': sample_school_holiday,
                'state_holiday': sample_state_holiday,
                'store_type': sample_store_type,
                'assortment': sample_assortment,
                'competition_distance': sample_competition_distance,
                'promo2': sample_promo2,
                'promo_interval': sample_promo_interval,
            })
            input_data = input_data.sort_values("date").reset_index(drop=True)
            st.session_state.input_data = input_data
            
            st.success("Sample data generated")
            
            # Show chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=input_data['date'],
                y=input_data['sales'],
                mode='lines',
                name='Sample Sales Data'
            ))
            fig.update_layout(
                title="Generated Sample Data",
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    # Streamlit reruns the script after button clicks. Keep the selected input
    # data available so clicking Run Prediction does not hide the forecast UI.
    if input_data is None and st.session_state.input_data is not None:
        input_data = st.session_state.input_data
    if input_data is not None:
        input_data = input_data.sort_values("date").reset_index(drop=True)
    
    # Prediction section
    if input_data is not None:
        st.markdown("---")
        st.header("Forecast Feature Inputs")

        future_template = future_feature_template(input_data, forecast_days)
        future_editor_key = (
            f"future_feature_editor_"
            f"{pd.to_datetime(input_data['date']).max().strftime('%Y%m%d')}_"
            f"{forecast_days}"
        )
        edited_future_features = st.data_editor(
            display_binary_labels(future_template),
            column_config=business_column_config(
                include_sales=False,
                disable_date=True,
            ),
            num_rows="fixed",
            use_container_width=True,
            key=future_editor_key,
        )
        future_features = normalize_editor_data(edited_future_features)

        st.header("Generate Forecast")
        
        # Center the button with empty columns on sides
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Run Prediction", type="primary", use_container_width=True, key="run_prediction"):
                with st.spinner("Generating forecast..."):
                    # Run prediction
                    results = st.session_state.predictor.predict(
                        input_data,
                        model_type=model_type,
                        forecast_days=forecast_days,
                        future_features=future_features,
                    )
                    
                    if results['success']:
                        st.success("Forecast generated successfully!")
                        
                        # Show metrics
                        st.markdown("### Forecast Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Total Forecast",
                                f"${results['summary']['total_predicted_sales']:,.0f}"
                            )
                        with col2:
                            st.metric(
                                "Daily Average",
                                f"${results['summary']['average_daily_sales']:,.0f}"
                            )
                        with col3:
                            st.metric(
                                "Forecast Period",
                                f"{forecast_days} days"
                            )
                        with col4:
                            st.metric(
                                "Model Used",
                                display_model_type(model_type)
                            )
                        
                        # Visualization
                        st.markdown("### Forecast Visualization")
                        
                        predictions_df = results['predictions']
                        historical_df = input_data.copy()
                        historical_df['date'] = pd.to_datetime(historical_df['date'])
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=historical_df['date'],
                            y=historical_df['sales'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=predictions_df['date'],
                            y=predictions_df['predicted_sales'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='green', width=3)
                        ))
                        
                        # Confidence interval
                        fig.add_trace(go.Scatter(
                            x=predictions_df['date'],
                            y=predictions_df['upper_bound'],
                            fill=None,
                            mode='lines',
                            line_color='rgba(0,255,0,0)',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=predictions_df['date'],
                            y=predictions_df['lower_bound'],
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(0,255,0,0.2)',
                            name='95% Confidence'
                        ))
                        
                        fig.update_layout(
                            title="Sales Forecast with Confidence Intervals",
                            xaxis_title="Date",
                            yaxis_title="Sales ($)",
                            hovermode='x unified',
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download section
                        st.markdown("### Export Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            # Prepare download data
                            export_df = predictions_df.copy()
                            export_df = export_df.round(2)
                            
                            csv = export_df.to_csv(index=False)
                            st.download_button(
                                label="Download Forecast (CSV)",
                                data=csv,
                                file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            st.info("Forecast includes predictions with confidence intervals")
                    
                    else:
                        st.error(f"Prediction failed: {results['error']}")

else:
    # No models loaded
    st.warning("Please load models using the sidebar before making predictions.")
    st.info("Click 'Load/Reload Models' in the sidebar to begin")
    
    # Add helpful information
    with st.expander("No models found? Here's what to do:", expanded=True):
        st.markdown("""
        ### First Time Setup
        
        If this is your first time using the system, you need to train the models:
        
        1. **Open Airflow UI**: [http://localhost:8080](http://localhost:8080)
           - Username: `admin`
           - Password: `admin`
        
        2. **Run the Training DAG**:
           - Find `sales_forecast_training` in the DAG list
           - Click the play button to trigger it
           - Wait for training to complete (5-10 minutes)
        
        3. **Come back here**:
           - Click "Load/Reload Models" again
           - Models should load successfully
        
        ### Quick Check
        
        - **MLflow UI**: [http://localhost:5001](http://localhost:5001) - Check if models exist
        - **MinIO UI**: [http://localhost:9001](http://localhost:9001) - Check artifact storage
          - Username: `minioadmin`
          - Password: `minioadmin`
        """)
