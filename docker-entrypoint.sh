#!/bin/bash
# docker-entrypoint.sh
# Starts either Streamlit UI or FastAPI based on the CMD argument

set -e

case "$1" in
    streamlit)
        echo "Starting Streamlit UI on port 8501 …"
        exec streamlit run app/streamlit_app.py \
            --server.port=8501 \
            --server.address=0.0.0.0 \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;
    api)
        echo "Starting FastAPI on port 8000 …"
        exec uvicorn api.fastapi_server:app \
            --host 0.0.0.0 \
            --port 8000
        ;;
    *)
        echo "Usage: docker run ... enterprise-rag [streamlit|api]"
        echo "  streamlit  — run the Streamlit UI (default)"
        echo "  api        — run the FastAPI server"
        exit 1
        ;;
esac
