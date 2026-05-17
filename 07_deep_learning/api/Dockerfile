FROM python:3.9-slim

WORKDIR /app

# Install numpy first — torch requires it at build time
RUN pip install --no-cache-dir "numpy<2"

# Install CPU-only torch (much smaller than the GPU version)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pillow python-multipart

# Copy model checkpoint and API code
COPY digit_cnn_checkpoint.pth .
COPY api/model.py .
COPY api/main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
