# 1. Use Python 3.10
FROM python:3.10

# 2. Set the working directory
WORKDIR /app

# 3. Copy the requirements file
COPY requirements.txt .

# 4. Install system build tools
RUN apt-get update && apt-get install -y build-essential

# 5. Install CPU-only PyTorch
RUN pip install torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# 6. Install the remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the application files
COPY . .

# 8. Expose the application port
EXPOSE 8000

# 9. Start the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
