NLP Movie Classifier
==========================

A production-grade NLP pipeline that classifies movie synopses into one of three tags:
- cult  
- dramatic  
- paranormal  

--------------------------
RUNNING VIA DOCKER (RECOMMENDED)
--------------------------
**1. Prerequisites**  
- Docker and Docker Compose installed  
- Folder `modelo_distilbert/` (with the trained model) must be in project root  

**2. Build the image:**

```bash
docker compose build --no-cache
```

**3. Start the API server:**

```bash
docker compose up
```

Access the API docs via Swagger UI at:  
http://localhost:8000/docs  

You can test:
- `/predict`: Send JSON with a synopsis  
- `/predict_file`: Upload a `.txt` file  

---

FEATURES
--------------------------
- End-to-end transformer-based pipeline (DistilBERT)
- Dataset cleaning, deduplication, and normalization
- Stratified train/test split
- Oversampling ONLY on training set to fix class imbalance (66% cult)
- FastAPI REST API
- Dockerized deployment for serving 
- Supports lightweight `--fast` mode for CPU-limited environments

--------------------------
FAST MODE vs FULL MODE
--------------------------
You can run the system in two ways:

**FAST MODE (`--fast`)**
- Developed for testing in CPU-only environments
- Uses a reduced dataset sample
- Trains for fewer steps (max_steps=50)
- Small batch size and only 1 epoch
- Useful for debugging or validating pipeline behavior without heavy compute

**FULL MODE (default)**
- Uses full dataset
- Full training regime with validation
- Recommended for actual model deployment

--------------------------
HOW TO RUN (LOCALLY)
--------------------------
1. Create virtual environment and install dev dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
source .venv/bin/activate           # Mac/Linux
pip install -r requirements-dev.txt
```

2. Run the full pipeline:

```bash
python main.py                     # full mode
python main.py --fast              # fast/debug mode
```

3. Run the API server locally:

```bash
uvicorn api:app --reload
```

Then open: http://localhost:8000/docs

--------------------------
FILES & STRUCTURE
--------------------------
├── src/  
│   ├── preprocessing.py       # Cleans and splits dataset  
│   ├── modeling.py            # Trains model  
│   ├── evaluate_model.py      # Model evaluation/report  
│   ├── utils.py               # Shared cleaning functions  
├── main.py                    # Pipeline runner (with --fast support)  
├── api.py                     # REST API using FastAPI  
├── requirements-api.txt       # Lightweight dependencies (Docker)  
├── requirements-dev.txt       # Full dependencies for training/evaluation  
├── Dockerfile                 # Image definition  
├── docker-compose.yml         # Full-stack launcher (API)  
├── modelo_distilbert/         # Saved trained model  
├── results/                   # Evaluation output, checkpoints  
└── data/                      # Cleaned and split CSV files  

--------------------------
REQUIREMENTS OVERVIEW
--------------------------
- `requirements.txt`:  
  Lightweight, for Docker and API runtime  

- `requirements-dev.txt`:  
  Full dependencies including: transformers, datasets, matplotlib, etc.  
  Required for training, EDA, evaluation  

--------------------------
TESTING THE API
--------------------------
Once the model is trained and API is running:

1. **POST /predict**  
   JSON Payload:  
   ```json
   { "text": "A mysterious figure appears in a haunted village..." }
   ```

2. **POST /predict_file**  
   Upload `.txt` file with synopsis

Test using: http://localhost:8000/docs  

--------------------------
AUTHOR
--------------------------
Ana Magalhães
