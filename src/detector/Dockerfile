FROM python:3.13.0-slim
WORKDIR /src/detector

# Set python path for models/
ENV PYTHONPATH=/src

# Install lib for PyVista
RUN apt-get update && apt-get install -y \
    libx11-6 \ 
    libxrender1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["fastapi", "run", "pipeline.py", "--port", "80"]