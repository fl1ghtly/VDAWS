FROM python:3.13.0-slim
WORKDIR /src/extractor

# Set python path to correctly find extractor module
ENV PYTHONPATH=/src

# Install libs for opencv-python-headless
RUN apt-get update && apt-get install -y \
    libxcb1 

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "extract.py"]