#!/bin/bash

PROJECT_ID="students-group2"
IMAGE_NAME="movie-recommender"
SERVICE_NAME="movie-recommender"
REGION="us-central1"

echo "=== 1. Configuration Docker ==="
gcloud auth configure-docker --quiet

echo "=== 2. Build Docker local ==="
cd ~/Fatma_Aziz/movie_recommender
docker build -t gcr.io/$PROJECT_ID/$IMAGE_NAME -f deploiement/Dockerfile .

echo "=== 3. Push vers GCR ==="
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME

echo "=== 4. Déploiement Cloud Run ==="
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
    --platform managed \
    --allow-unauthenticated \
    --region $REGION \
    --memory 1Gi \
    --cpu 1

echo "=== 5. URL du service ==="
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)")
echo "URL: $SERVICE_URL"