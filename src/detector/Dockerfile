FROM python:3.13.0-slim
WORKDIR /src/detector

# Set python path for models/
ENV PYTHONPATH=/src

# Install libs for PyVista and headless 3D rendering
RUN apt-get update && apt-get install -y \
    libx11-6 \ 
    libxrender1 \
    libgl1-mesa-glx \
    libegl1 \
    libosmesa6 \
    libxcursor1 \
    libglib2.0-0 \
    xvfb \ 
    && rm -rf /var/lib/apt/lists/*  # clean cache for smaller image size

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["fastapi", "run", "pipeline.py", "--port", "80"]