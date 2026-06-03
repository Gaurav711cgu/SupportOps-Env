FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY env/ ./env/
COPY server.py .
COPY openenv.yaml .
COPY inference.py .
COPY README.md .

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
```
Save and close.

---

**Step 2 — Edit requirements.txt:**
```
notepad requirements.txt
```

Replace everything with:
```
fastapi==0.110.0
uvicorn==0.29.0
pydantic==2.6.4
openai==1.30.0
requests==2.31.0
python-multipart==0.0.9
httpx==0.27.0
```
Save and close.

---

**Step 3 — Push:**
```
git add Dockerfile requirements.txt
git commit -m "Fix Dockerfile and requirements"
git push