FROM python:3.11-slim

# LightGBM requires libgomp (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy just pyproject.toml first for dependency layer caching
COPY pyproject.toml .

# Create minimal package structure so pip install works
RUN mkdir -p demandops && touch demandops/__init__.py

# Install dependencies (cached unless pyproject.toml changes)
RUN pip install --no-cache-dir .

# Now copy actual source
COPY demandops/ demandops/
COPY configs/ configs/

EXPOSE 8001

CMD ["uvicorn", "demandops.serving.app:app", "--host", "0.0.0.0", "--port", "8001"]
