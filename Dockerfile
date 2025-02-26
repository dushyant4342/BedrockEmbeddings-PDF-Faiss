# Use Amazon Linux 2023 as the base image
FROM amazonlinux:2023

# Set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Install dependencies
RUN yum update -y && \
    yum install -y python3 python3-pip && \
    pip3 install --upgrade pip

# Set working directory inside the container
WORKDIR /app

# Copy only requirements.txt first (so Docker caches dependencies)
COPY requirements.txt .

# Install Python dependencies first (cached layer)
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir faiss-cpu

# Copy the rest of the application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
