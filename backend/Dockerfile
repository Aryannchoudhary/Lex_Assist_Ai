# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# 1. Upgrade pip to the latest, most network-resilient version first
RUN pip install --upgrade pip

# 2. Install requirements with extreme patience and caching enabled
RUN pip install --default-timeout=2000 --retries=20 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# Copy the rest of the backend code into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]