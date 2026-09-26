FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and ensure logs appear immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Tesseract OCR
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app ./app
COPY src ./src
COPY data ./data
COPY models ./models

# Streamlit port
EXPOSE 8501

# Start Streamlit
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]