FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Force CPU-only torch BEFORE installing anything else
# This prevents pip from pulling 288MB CUDA packages
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers after torch is already CPU
RUN pip install --no-cache-dir \
    sentence-transformers==3.0.1

# Install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/faiss_index data/clean_text data/raw_html data/pdfs pdf_data/files

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]