# Start from a small official Python image (slim = lightweight).
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy and install dependencies first. Doing this before copying the rest
# lets Docker cache this layer, so rebuilds are faster when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the trained model artifact into the image.
COPY app.py model.pkl ./

# Document that the container listens on port 8000.
EXPOSE 8000

# The command that runs when the container starts: launch the API server.
# host 0.0.0.0 makes it reachable from outside the container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
