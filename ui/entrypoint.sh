#!/usr/bin/env bash
set -e

exec streamlit run /app/app.py --server.address=0.0.0.0 --server.port=8501
