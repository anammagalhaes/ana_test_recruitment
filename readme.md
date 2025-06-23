NLP Movie Classifier
==========================

A production-grade NLP pipeline that classifies movie synopses into one of three tags:
  - cult
  - dramatic
  - paranormal

--------------------------
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

FAST MODE (`--fast`)
  - Developed for testing in CPU-only environments
  - Uses a reduced dataset sample
  - Trains for fewer steps (max_steps=50)
  - Small batch size and only 1 epoch
  - Useful for debugging or validating pipeline behavior without heavy compute

FULL MODE (default)
  - Uses full dataset
  - Full training regime with validation
  - Recommended for actual model deployment

--------------------------
HOW TO RUN (LOCALLY)
--------------------------
1. Create virtual environment and install dev dependencies:
   python -m venv .venv
   .venv\Scripts\activate              # Windows
   source .venv/bin/activate           # Mac/Linux
   pip install -r requirements-dev.txt

2. Run the full pipeline:
   python main.py                     # full mode
   python main.py --fast              # fast/debug mode

3. Run the API server:
   uvicorn api:app --reload
   Access API docs at: http://localhost:8000/docs

--------------------------
HOW TO RUN (DOCKER)
--------------------------
To launch everything in production setup:

   docker compose up --build

After build:
- FastAPI available at http://localhost:8000/docs

--------------------------
FILES & STRUCTURE
--------------------------
├── src/
│   ├── preprocessing.py       # Cleans and splits dataset
│   ├── modeling.py            # Trains model
│   ├── evaluate_model.py      # Model evaluation/report
│   ├── utils.py               # Shared cleaning functions
│   └── log_info.py            # Logs training arguments and metadata
├── main.py                    # Pipeline runner (with --fast support)
├── api.py                     # REST API using FastAPI
├── requirements.txt           # Lightweight dependencies (Docker)
├── requirements-dev.txt       # Full dependencies for training/evaluation
├── Dockerfile                 # Image definition
├── docker-compose.yml         # Full-stack launcher (API + UI)
├── modelo_distilbert/         # Saved trained model
├── results/                   # Evaluation output, checkpoints
└── data/                      # Cleaned and split CSV files

--------------------------
REQUIREMENTS OVERVIEW
--------------------------
- requirements.txt:
    Lightweight for Docker & API usage only

- requirements-dev.txt:
    Includes transformers, datasets, matplotlib, etc.
    Required for full training/evaluation locally

--------------------------
TESTING THE API
--------------------------
After model is trained and API is running:

1. POST /predict
   Payload: { "text": "..." }

2. POST /predict_file
   Upload .txt file with synopsis text

--------------------------
AUTHOR
--------------------------
Ana Magalhães

