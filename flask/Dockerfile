FROM ubuntu:latest
LABEL authors="dhruv"

ENTRYPOINT ["top", "-b"]

# Use the official Python image as the base image
FROM python:3.11 as base

# Set the working directory in the container
WORKDIR /app

# Update the package list and install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Intermediate image with just the installed dependencies
FROM base as intermediate

# Copy the entire Flask project into the container
COPY . .

# Final image with the application code
FROM base

# Copy the application code from the intermediate image
COPY --from=intermediate /app /app

# Command to run Flask application with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--access-logfile", "-", "--error-logfile", "-", "run:app"]

