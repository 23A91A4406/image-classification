FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY model ./model
COPY data ./data

EXPOSE ${API_PORT}

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
