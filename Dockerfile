FROM python:3.11-slim

WORKDIR /app

# System libs needed by rasterio + opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy project
COPY . .

# Generate sample data + train if no checkpoint present
RUN python setup_model.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
