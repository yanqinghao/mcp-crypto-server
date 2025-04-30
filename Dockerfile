# Dockerfile

# ---- Build Stage ----
# Use a full Python image that includes build tools
FROM python:3.12 as builder

# Install build dependencies for TA-Lib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib C library
WORKDIR /tmp
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4/ && \
   ./configure --prefix=/usr && \
    make && \
    make install

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Final Stage ----
# Use the slim Python image for the final, smaller image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Copy required system libraries installed in the builder stage (TA-Lib.so files)
# Find the exact location if needed, usually in /usr/lib or /usr/local/lib
COPY --from=builder /usr/lib/libta-lib.* /usr/lib/

# Copy the virtual environment from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copy application code
COPY . /app/

# Create a non-root user and group [1, 2, 3, 4, 5, 6, 7, 8, 9]
RUN addgroup --system app && adduser --system --group app

# Switch to the non-root user
USER app

# Expose the application port
EXPOSE 8000

# Define the default command to run the application
# Ensure 'your_app_entrypoint.py' is the correct entry point for your app
# For the example server, it would be 'main.py' if it's in the root,
# or adjust the path if it's inside the 'app' directory copied earlier.
# Assuming main.py is in the root of the context copied to /app
CMD ["python", "main.py"]
