# 1️⃣ Image Python officielle (légère et compatible Cloud Run)
FROM python:3.10-slim

# 2️⃣ Variables d’environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cloud Run écoute sur 8080
ENV PORT=8080

# 3️⃣ Dossier de travail
WORKDIR /app

# 4️⃣ Installer les dépendances système minimales
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5️⃣ Copier requirements et installer les libs Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copier tout le projet
COPY . .

# 7️⃣ Exposer le port Cloud Run
EXPOSE 8080

# 8️⃣ Lancer FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
