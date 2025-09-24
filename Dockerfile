# Use a Python 3.10 base image
FROM python:3.10-slim-bullseye

# Set the working directory
WORKDIR /usr/src/app

# Install system dependencies for scientific packages
RUN apt-get update && apt-get install -y \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your app will run on
EXPOSE 10000

# Start your application with Gunicorn
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:10000", "app:app"]