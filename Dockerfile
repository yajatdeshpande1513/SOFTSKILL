# Use a specific Python version as the base image.
# We're using a slim version to keep the image small.
FROM python:3.11.8-slim-bullseye

# Set the working directory inside the container.
WORKDIR /usr/src/app

# Install system dependencies for scientific packages like scipy and scikit-learn.
# 'gfortran' is the Fortran compiler that the build was failing without.
# 'libopenblas-dev' and 'liblapack-dev' are linear algebra libraries.
RUN apt-get update && apt-get install -y \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container.
COPY requirements.txt ./

# Install Python packages from the requirements file.
# The --no-cache-dir flag helps keep the image size down.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container.
COPY . .

# Expose the port that your Flask application will run on.
EXPOSE 10000

# Define the command to start your application when the container launches.
# We use the Gunicorn web server for a production environment.
# app:app refers to the Flask instance named 'app' in the 'app.py' file.
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:10000", "app:app"]