# Use a Python 3.10 base image which is compatible with TensorFlow 2.15.0
FROM python:3.10-buster

ARG BUILD_ENV
ARG OVERCOOKED_BRANCH
ARG GRAPHICS

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by overcooked_ai or its dependencies
# Added git as it's needed for cloning/installing overcooked_ai[harl]
# Added libgl1-mesa-dev as seen in your original Dockerfile (might be needed for rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-dev \
    # Add any other system dependencies needed by requirements.txt or overcooked_ai
    && rm -rf /var/lib/apt/lists/*

# Install non-chai dependencies
# Copy requirements.txt before installing to leverage Docker caching
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install eventlet production server if production build
# Note: Your up.sh script sets BUILD_ENV=production, so this will always run with that script.
RUN if [ "$BUILD_ENV" = "production" ] ; then pip install --no-cache-dir eventlet ; fi

# --- Fix for Python 3.10 collections.MutableMapping Error ---
# Explicitly install Python 3.10+ compatible versions of eventlet and dnspython
# This is necessary because older versions are not compatible with Python 3.10+
# and might be pulled in by dependencies.
RUN pip install --no-cache-dir "eventlet>=0.33.0" "dnspython>=2.0.0"
# --- End Fix ---


# Clone chai code
# Using --depth 1 to reduce clone size/time if you don't need full history
RUN git clone --recursive --depth 1 https://github.com/HumanCompatibleAI/overcooked_ai.git --branch $OVERCOOKED_BRANCH --single-branch /overcooked_ai

# Dummy data_dir so things don't break (as in original Dockerfile)
# Ensure the path within the container is correct relative to where overcooked_ai is cloned
RUN echo "import os; DATA_DIR=os.path.abspath('/overcooked_ai/src/human_aware_rl')" >> /overcooked_ai/src/human_aware_rl/data_dir.py

# Install chai dependencies in editable mode
# The [harl] extra requires git, which is installed above.
RUN pip install --no-cache-dir -e '/overcooked_ai[harl]'

# Copy over remaining application files
# *** IMPORTANT: Ensure these files/directories exist in your build context (where you run docker compose build) ***
COPY ./static ./static
COPY ./*.py ./
COPY ./graphics/$GRAPHICS ./static/js/graphics.js
COPY ./config.json ./config.json

# Clean up apt cache and temporary files
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/* \
    && rm -rf /tmp/*

# Set environment variables that will be used by app.py
ENV HOST=0.0.0.0
ENV PORT=5000
ENV CONF_PATH=config.json

# Expose the port the application listens on
EXPOSE 5000

# Command to run when the container starts
# This assumes the main application script is named app.py in the WORKDIR (/app)
CMD ["python", "-u", "app.py"]
