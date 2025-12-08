FROM python:3.13.0-slim
WORKDIR /extractor
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "extractor.py"]