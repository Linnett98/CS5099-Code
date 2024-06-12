FROM nvcr.io/nvidia/pytorch:23.05-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . /app

# Set the working directory
WORKDIR /app/src/data-processing

# Command to run your script
CMD ["python", "data-processing.py"]
