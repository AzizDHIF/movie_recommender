#!/bin/bash

# Définir l'ID du projet GCP
PROJECT_ID="students-group2"

# Nom de l'image et du service
IMAGE_NAME="movie-recommender"
SERVICE_NAME="movie-recommender"
REGION="us-central1"

echo "=== Build de l'image Docker ==="
# CHANGER : Aller au dossier parent et builder depuis là
cd ..
gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME .

echo "=== Déploiement sur Cloud Run ==="
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME \
    --platform managed \
    --allow-unauthenticated \
    --region $REGION \
    --memory 1Gi \
    --cpu 1

echo "=== Déploiement terminé ! ==="