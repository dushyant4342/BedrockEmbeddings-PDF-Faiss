# Use Amazon Linux 2023 as base image
FROM amazonlinux:2023

# Set working directory
WORKDIR /app

# Install Python 3.9 and pip
RUN yum update -y && \
    yum install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \ 
    ln -sf /usr/bin/pip3 /usr/bin/pip  

# (ln -s symbolic link which can run pip install instead of pip3 install.)

# Copy files from GitHub repo
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]