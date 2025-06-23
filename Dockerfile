FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Define diretório de trabalho
WORKDIR /app

# Atualiza pacotes e instala Git 
RUN apt-get update && apt-get install -y git

# Copia todos os arquivos para o container
COPY . /app

# Atualiza pip e instala dependências da API
RUN pip install --upgrade pip
RUN pip install -r requirements-api.txt

# Expõe a porta usada pelo Uvicorn
EXPOSE 8000

# Inicia o servidor FastAPI com hot reload desabilitado (produção)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
