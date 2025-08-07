FROM python:3.9-slim

WORKDIR /mlflow_server

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Default command
CMD ["mlflow", "server", "--help"]