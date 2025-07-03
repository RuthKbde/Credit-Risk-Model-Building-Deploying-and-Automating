# Dockerfile

# Use an official Python runtime as a parent image
# Using a specific version is good practice for reproducibility
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Prevents pip from caching downloaded packages, reducing image size
# -r: Install from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container at /app
# This includes src/, tests/, pytest.ini, etc.
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application using uvicorn
# 0.0.0.0: Tells uvicorn to listen on all available network interfaces
# 8000: The port to listen on
# src.api.main: Refers to the main.py file inside the src/api directory
# --host 0.0.0.0 --port 8000: Explicitly sets host and port for uvicorn
# --reload: REMOVE THIS FOR PRODUCTION! Good for development, but not production builds.
#           It watches for code changes and reloads, which adds overhead.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]