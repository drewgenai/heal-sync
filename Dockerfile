# Get a distribution that has uv already installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install required system dependencies for building packages
USER root
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    python3-dev \
    libssl-dev \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add user - this is the user that will run the app
RUN useradd -m -u 1000 user
USER user

# Set the home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH        

ENV UVICORN_WS_PROTOCOL=websockets

# Set the working directory
WORKDIR $HOME/app

# Copy the app to the container
COPY --chown=user . $HOME/app

# Install the dependencies
RUN uv sync

# Ensure NLTK is available inside the uv virtual environment and download required data
RUN uv pip install nltk \
    && uv run python3 -m nltk.downloader punkt punkt_tab

# Expose the port
EXPOSE 7860

# Run the app
CMD ["uv", "run", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
