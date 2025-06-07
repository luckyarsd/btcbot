# Use a specific Python base image
FROM python:3.10-slim-buster

# Install system dependencies required for TA-Lib and others
# apt-get update: Updates the package list
# apt-get install -y --no-install-recommends: Installs packages without recommended dependencies
#   build-essential: For compiling C extensions (like TA-Lib)
#   libta-lib-dev: The actual TA-Lib C library development files
#   git: Often useful for cloning repos, though not strictly needed by your app
#   tzdata: For timezone data (pytz dependency)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libta-lib-dev \
    git \
    tzdata && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure your bot.py is executable (optional but good practice)
# RUN chmod +x bot.py

# Define the command to run your application
# This replaces the Procfile for Docker deployments
CMD ["python", "bot.py"]
