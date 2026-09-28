FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN python -m pytest tests/ -q -W ignore::UserWarning || exit 1

ENTRYPOINT ["python", "-m", "banditforge.cli"]
CMD ["benchmark", "--output", "benchmark.json"]
