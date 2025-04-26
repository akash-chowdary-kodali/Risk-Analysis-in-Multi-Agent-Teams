FROM ubuntu:22.04

WORKDIR /app

COPY ./requirements.txt ./requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libevent-dev \
    pkg-config \
    gcc \
    libc6-dev \
    libffi-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
RUN tar -xf Python-3.10.13.tgz
RUN cd Python-3.10.13 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -rf Python-3.10.13*

# Configure virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN /opt/venv/bin/pip install --upgrade pip wheel

# Install Python dependencies
RUN CFLAGS="-fPIC" \
    LDFLAGS="-L/usr/lib/x86_64-linux-gnu" \
    /opt/venv/bin/pip install --no-cache-dir --no-build-isolation \
    -r requirements.txt

RUN if [ "production" = "production" ] ; then /opt/venv/bin/pip install "psycopg2-binary"; fi

# Clone the overcooked_ai repository
RUN git clone --recursive https://github.com/HumanCompatibleAI/overcooked_ai /app/overcooked_ai

# Set data directory (Corrected path)
RUN mkdir -p /app/overcooked_ai/overcooked_ai && \
    echo "import os; DATA_DIR=os.path.abspath('./overcooked_ai')" >> /app/overcooked_ai/overcooked_ai/__init__.py

# Install Overcooked AI
RUN /opt/venv/bin/pip install -e '/app/overcooked_ai[harl]'

# Copy the application files (Corrected path)
COPY ./src /app/src

# Clean up
RUN apt-get purge -y --auto-remove \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/*

# Add CMD or ENTRYPOINT here to run your application
# Example: Assuming your main application script is named "app.py"
CMD ["python", "/app/src/overcooked_demo/app.py"]
