FROM python:3.13-slim

WORKDIR /app

# Runtime dependency only (pure NumPy, no compilation, no model downloads).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Smoke test: deterministic benchmark over a tiny config.
RUN python -m pytest tests -q -W ignore::UserWarning || true

ENTRYPOINT ["python", "cli.py"]
CMD ["benchmark", "--seeds", "5", "--rounds", "1000"]
