# Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

# Optional: Train model during build
RUN python train_model.py

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
