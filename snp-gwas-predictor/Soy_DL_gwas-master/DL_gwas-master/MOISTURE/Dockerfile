FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy your app and dependency list into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Use interactive terminal output
ENV PYTHONUNBUFFERED=1

# Default command (can override in docker run)
CMD ["python3", "run_all_folds.py"]
