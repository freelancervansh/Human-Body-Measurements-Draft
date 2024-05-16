FROM python:3.10
COPY . . 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
WORKDIR /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]