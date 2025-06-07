# Use a Python 3.10 slim image based on Debian Bullseye
FROM python:3.10-slim-bullseye

# Install system dependencies required for TA-Lib and others
# apt-get update: Updates the package list
# apt-get install -y --no-install-recommends: Installs packages without recommended dependencies
#   build-essential: For compiling C extensions (like TA-Lib)
#   libta-lib-dev: The actual TA-Lib C library development files (hopefully found now)
#   git: Often useful for cloning repos
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

# Define the command to run your application
CMD ["python", "bot.py"]
