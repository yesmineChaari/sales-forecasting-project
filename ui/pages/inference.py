"""
Model Inference Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.simple_model_loader import SimpleModelLoader
from utils.simple_predictor import SimplePredictor

# Initialize session state
if 'model_loader' not in st.session_state:
    st.session_state.model_loader = SimpleModelLoader()
    st.session_state.predictor = SimplePredictor(st.session_state.model_loader)
    st.session_state.models_loaded = False
    st.session_state.run_id = None

st.header("🔮 Model Inference")

# Model loading section
with st.expander("📦 Model Management", expanded=not st.session_state.models_loaded):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not st.session_state.models_loaded:
            st.warning("⚠️ No models loaded. Click 'Load Latest Models' to begin.")
        else:
            st.success(f"✅ Models loaded from run: {st.session_state.run_id[:8] if st.session_state.run_id else 'Unknown'}...")
            st.info(f"Models: {', '.join(st.session_state.model_loader.models.keys())}")
    
    with col2:
        if st.button("🔄 Load Latest Models", type="primary"):
            with st.spinner("Loading models from MLflow..."):
                # Get latest run
                run_id = st.session_state.model_loader.get_latest_run()
                if not run_id:
                    st.error("No trained models found in MLflow. Please train a model first.")
                
                if run_id and st.session_state.model_loader.load_models_from_run(run_id):
                    st.session_state.models_loaded = True
                    st.session_state.run_id = run_id
                    st.success("✅ Models loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load models. Check MLflow connection.")

# Inference section
if st.session_state.models_loaded:
    st.subheader("📊 Generate Predictions")
    
    # Input configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        store_id = st.selectbox(
            "Store ID",
            ["store_001", "store_002", "store_003", "All Stores"],
            help="Select store for prediction"
        )
    
    with col2:
        model_type = st.selectbox(
            "Model Type",
            ["ensemble", "xgboost", "lightgbm"],
            help="Select model for prediction"
        )
    
    with col3:
        forecast_days = st.number_input(
            "Forecast Days",
            min_value=1,
            max_value=90,
            value=30,
            help="Number of days to forecast"
        )
    
    # Data input section
    st.subheader("📥 Input Data")
    
    input_method = st.radio(
        "Select input method:",
        ["Use Sample Data", "Upload CSV", "Manual Entry"],
        horizontal=True
    )
    
    input_data = None
    
    if input_method == "Use Sample Data":
        # Generate sample data
        if st.button("Generate Sample Data"):
            dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
            sample_data = pd.DataFrame({
                'date': dates,
                'store_id': store_id if store_id != "All Stores" else "store_001",
                'sales': np.random.normal(5000, 1000, len(dates)).clip(0)
            })
            input_data = sample_data
            st.session_state.input_data = input_data
            st.success("✅ Sample data generated")
            
            with st.expander("View Sample Data"):
                st.dataframe(sample_data.tail(10))
    
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload historical sales data",
            type=['csv'],
            help="CSV should contain: date, store_id (optional), sales"
        )
        
        if uploaded_file is not None:
            input_data = pd.read_csv(uploaded_file)
            st.session_state.input_data = input_data
            st.success(f"✅ Loaded {len(input_data)} records")
            
            with st.expander("View Uploaded Data"):
                st.dataframe(input_data.head(10))
    
    elif input_method == "Manual Entry":
        st.info("Enter recent sales data (last 7 days)")
        
        manual_data = []
        cols = st.columns(7)
        
        for i in range(7):
            date = datetime.now() - timedelta(days=6-i)
            with cols[i]:
                sales = st.number_input(
                    f"{date.strftime('%m/%d')}",
                    min_value=0,
                    value=5000,
                    key=f"manual_{i}"
                )
                manual_data.append({
                    'date': date,
                    'store_id': store_id if store_id != "All Stores" else "store_001",
                    'sales': sales
                })
        
        input_data = pd.DataFrame(manual_data)
        # Store in session state to persist across reruns
        st.session_state.input_data = input_data
    
    # Check for input_data in session state if not set locally
    if 'input_data' not in locals() and 'input_data' in st.session_state:
        input_data = st.session_state.input_data
    
    # Generate predictions
    if input_data is not None:
        st.markdown("---")
        # Center the generate button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_clicked = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
        
        if generate_clicked:
            with st.spinner("Generating predictions..."):
                results = st.session_state.predictor.predict(
                    input_data,
                    model_type=model_type,
                    forecast_days=forecast_days
                )
                
                if results['success']:
                    st.success("✅ Forecast generated successfully!")
                    
                    # Display summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Predicted Sales",
                            f"${results['summary']['total_predicted_sales']:,.0f}"
                        )
                    with col2:
                        st.metric(
                            "Average Daily Sales",
                            f"${results['summary']['average_daily_sales']:,.0f}"
                        )
                    with col3:
                        st.metric(
                            "Forecast Period",
                            f"{results['summary'].get('forecast_days', forecast_days)} days"
                        )
                    
                    # Visualization
                    st.subheader("📈 Forecast Visualization")
                    
                    predictions_df = results['predictions']
                    historical_df = input_data.copy()
                    historical_df['date'] = pd.to_datetime(historical_df['date'])
                    
                    fig = go.Figure()
                    
                    # Historical data
                    if 'sales' in input_data.columns:
                        fig.add_trace(go.Scatter(
                            x=historical_df['date'],
                            y=historical_df['sales'],
                            mode='lines+markers',
                            name='Historical Sales',
                            line=dict(color='blue', width=2),
                            marker=dict(size=6)
                        ))
                
                # Predictions
                    fig.add_trace(go.Scatter(
                        x=predictions_df['date'],
                        y=predictions_df['predicted_sales'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='green', width=3),
                        marker=dict(size=6)
                    ))
                
                    # Confidence intervals
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
                        line_color='rgba(0,255,0,0)',
                        name='Confidence Interval'
                    ))
                
                    fig.update_layout(
                        title="Sales Forecast",
                        xaxis_title="Date",
                        yaxis_title="Sales ($)",
                        hovermode='x unified',
                        height=500
                    )
                
                    st.plotly_chart(fig, use_container_width=True)
                
                    # Model comparison if ensemble
                    model_predictions = results.get('model_predictions', {})
                    if model_type == "ensemble" and len(model_predictions) > 1:
                        st.subheader("📊 Model Comparison")
                    
                        comparison_data = []
                        for model_name, preds in model_predictions.items():
                            if model_name != 'ensemble':
                                preds = np.asarray(preds)
                                comparison_data.append({
                                    'Model': model_name,
                                    'Average Prediction': np.mean(preds),
                                    'Min': np.min(preds),
                                    'Max': np.max(preds)
                                })
                    
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df)
                
                    # Download results
                    st.subheader("💾 Download Results")
                
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv,
                        file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.error(f"❌ Prediction failed: {results['error']}")

else:
    st.info("👆 Please load models first to begin making predictions.")
