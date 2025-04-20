# Base Python image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy dependency files first for Docker caching
COPY requirements.txt requirements.txt
COPY setup.py setup.py

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Flask app port
EXPOSE 8080

# Command to run the Flask application
CMD ["python", "app.py"]
