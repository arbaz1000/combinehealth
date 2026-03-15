#!/bin/bash
# Start both FastAPI backend and Streamlit frontend
# FastAPI runs on port 8000 (internal), Streamlit on port 7860 (exposed)

# Start FastAPI in the background
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit (foreground — this is the main process)
streamlit run app/frontend.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false