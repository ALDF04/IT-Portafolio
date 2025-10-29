#!/usr/bin/env bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboards/app_streamlit.py
