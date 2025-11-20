FROM python:3.11-slim

# Make stdout/err unbuffered for cleaner logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . .

# Run the combined short screener + opener
CMD ["python", "main.py"]
