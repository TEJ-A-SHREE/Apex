FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all environment files
COPY . .

# Expose port for HuggingFace Spaces
EXPOSE 7860

# Environment variables (override at runtime)
ENV PORT=7860
ENV DIFFICULTY=easy
ENV PYTHONPATH=/app

# Start the server
CMD ["python", "server.py"]
