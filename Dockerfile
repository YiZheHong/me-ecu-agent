# Dockerfile for me-ecu-agent
# This creates a base image with the ECU agent library installed

FROM python:3.10-slim

LABEL maintainer="Potter Hong"
LABEL description="ECU Agent - Document retrieval and RAG system"

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for certain Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better caching)
COPY setup.py .
COPY requirements.txt* ./

# Copy source code
COPY src/ ./src/

# Install the package in editable mode
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -e .

# Verify installation
RUN python -c "from me_ecu_agent.query import QueryFactory; print('✅ me-ecu-agent installed successfully')"

# Default command (can be overridden)
CMD ["python"]