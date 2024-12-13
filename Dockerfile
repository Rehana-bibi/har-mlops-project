FROM python:3.8

WORKDIR /mlflow_project

# Install system dependencies for matplotlib, scikit-learn, and git
RUN apt-get update && apt-get install -y \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p mlruns