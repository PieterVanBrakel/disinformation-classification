FROM python:3.11-slim

WORKDIR /app

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY pyproject.toml ./

RUN pip install uv && uv pip install --system fastapi uvicorn pydantic scikit-learn nltk pyyaml

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
