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
    # Dependencias de sistema para Chromium (Debian Trixie)
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2t64 \
    libpango-1.0-0 \
    libcairo2 \
    fonts-liberation \
    fonts-unifont \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para cachear dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Instalar Chromium sin --with-deps (dependencias ya instaladas arriba)
RUN playwright install chromium

# Copiar código del proyecto
COPY . .

# Exponer puerto FastAPI
EXPOSE 8000

# Comando de inicio con autoreload para debug (puedes quitar --reload en prod)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
