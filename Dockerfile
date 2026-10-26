# Serving image: ONNX Runtime only, no PyTorch (~250 MB instead of ~2 GB).
FROM python:3.11-slim
WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY petseg/__init__.py petseg/utils.py petseg/inference.py petseg/
COPY app app
COPY models/petseg.onnx models/
ENV PETSEG_MODEL=/srv/models/petseg.onnx PETSEG_THREADS=4
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
