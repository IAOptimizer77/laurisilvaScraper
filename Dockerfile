# ===============================
# 🐍 FASTAPI + OPENAI + QDRANT
# ===============================
FROM python:3.11-slim

# Evitar prompts interactivos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para cachear dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Instalar Chromium y sus dependencias de sistema para Playwright
RUN playwright install chromium --with-deps

# Copiar código del proyecto
COPY . .

# Exponer puerto FastAPI
EXPOSE 8000

# Comando de inicio con autoreload para debug (puedes quitar --reload en prod)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
