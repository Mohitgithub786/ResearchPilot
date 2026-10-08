FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any are needed for chroma/sqlite
RUN apt-get update && apt-get install -y build-essential curl

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860
ENV FASTAPI_BASE_URL=http://127.0.0.1:7860

# Expose port for Hugging Face / Render
EXPOSE 7860

# Command to run the unified FastAPI + Gradio app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]