# Dockerfile:

# Use a base image with Python
FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY requirements.txt requirements.txt
COPY model.py model.py

# Install the required Python libraries
RUN pip install -r requirements.txt

# Run the Python script when the container starts
CMD ["python", "model.py"]