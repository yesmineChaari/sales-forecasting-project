#!/bin/bash
set -e

streamlit run inference_app.py --server.address 0.0.0.0 --server.port 8501
