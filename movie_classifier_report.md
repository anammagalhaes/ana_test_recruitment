Movie Classifier - Report
=========================================

Project Summary
---------------
This project is an end-to-end NLP pipeline for classifying movie synopses into one of three genres: "cult", "dramatic", or "paranormal". The system leverages a transformer-based architecture using DistilBERT, offering a REST API and easy deployment through Docker.

It was developed considering training time, lack of GPU, prediction quality, and code scalability for future improvements. This report goes through each developed step, prediction results, best usage practices, and a suggested backlog. The README complements this with the structure of the modules and instructions to run the code both locally and via Docker and API.

Pipeline Stages
---------------
1. Exploratory Data Analysis (EDA):
   - Script: 01_EDA_TASK.ipynb
   - Corrupted CSV and input treated as one string per line
   - Data validation and cleaning: removal of special characters, duplicate synopses, NaNs, lowercase standardization
   - Word count statistics by genre
   - Synopses had varied lengths, which led to truncation and padding later
   - Label distribution showed imbalance (cult > 65%)
   - Semantic complexity guided the decision toward transformers rather than manual feature extraction + classifiers

2. Preprocessing & Balancing:
   - Script: src/preprocessing.py
   - Stratified train/test split with 90:10 ratio
   - Oversampling applied only to training data to mitigate class imbalance
   - The same cleaning function is reused for user input at inference time

3. Model Training:
   - Script: src/modeling.py
   - Transformer: DistilBERT (fine-tuned), lightweight and suitable for CPU environments
   - LabelEncoder used to convert text labels into numeric format
   - Tokenization using DistilBertTokenizerFast with automatic truncation and padding
   - Training via HuggingFace's Trainer and TrainingArguments
   - Evaluation strategy set to "epoch" with load_best_model_at_end=True
   - Metric of choice: macro-F1, to give equal importance to all classes

4. Model Evaluation:
   - Script: src/evaluate_model.py
   - Classification report printed in terminal
   - Predictions saved in results/predictions.csv
   - Confusion matrix and class-wise metrics included

5. Serving the Model:
   - Script: api.py using FastAPI
   - Endpoints: /predict and /predict_file
   - Swagger UI available at http://localhost:8000/docs
   - Supports both .txt file upload and direct JSON input

6. Docker Support:
   - Uses Dockerfile and docker-compose.yml
   - Two requirements files:
     - requirements.txt (runtime only)
     - requirements-dev.txt (for full training and EDA)

FAST vs FULL Mode
-----------------
FAST Mode (--fast in main.py)
- Designed for low-resource (CPU-only) environments
- Uses small batch size, max 1 epoch, max_steps = 50
- Useful for testing and debugging pipeline logic

FULL Mode (default)
- Uses the full dataset
- Full training regime with validation
- Suitable for production-quality models

Model Justification
-------------------
DistilBERT was chosen as a lightweight yet performant transformer model that balances speed and accuracy, especially suitable under compute constraints.

Other options like SBERT or embedding + classifiers (e.g., logistic regression) were considered but discarded due to:
- Lack of task-specific fine-tuning
- Inferior performance during preliminary experiments
- DistilBERT offers end-to-end, context-sensitive learning

Still, simpler baselines are recommended as future benchmarks (see backlog).

Logging & Checkpoints
---------------------
- Training checkpoints are saved in /results/
- Trained model and tokenizer are stored in /modelo_distilbert/
- Training hyperparameters can be viewed via log_info.py

Evaluation Results
------------------
Classification Report (printed in terminal):

              precision    recall  f1-score   support
        cult     0.8699    0.6195    0.7236       205
    dramatic     0.3396    0.5455    0.4186        33
  paranormal     0.5091    0.7887    0.6188        71

    accuracy                         0.6505       309
   macro avg     0.5729    0.6512    0.5870       309
weighted avg     0.7303    0.6505    0.6670       309

Confusion Matrix:

[[127  28  50]
 [ 11  18   4]
 [  8   7  56]]

Prediction Output and Discussion
------------------
- The file results/predictions.csv contains all predictions and their true labels
- The evaluation was based on the full dataset (data/task.csv) after cleaning, splitting, and balancing
- Final test set was saved as data/test_data.csv
- The model was trained in full mode using default training parameters
- The "cult" class achieved high precision, but recall was low — many were misclassified as "paranormal"
- The "dramatic" class showed the weakest performance, likely due to low representation and overlapping features
- Despite imbalance, the macro-F1 score reached 0.5870
- Final test accuracy was 65.05%, which is a reasonable baseline

Best Practices for Usage
-------------------------
To use the model via the API:

1. Prepare a .txt file with a single movie synopsis  
   Example:
   A troubled teenager discovers she has telekinetic powers after moving to a new town haunted by local legends.

2. Open Swagger UI in your browser:  
   http://localhost:8000/docs

3. Use the /predict_file endpoint to upload the .txt file and submit the request

4. Alternatively, use the /predict endpoint by submitting raw JSON input:
   {
     "text": "A story of a cursed painting that haunts everyone who owns it."
   }

Backlog & Improvements
-----------------------
- Jupyter notebook for analyzing predictions (planned, not delivered)
- Refactor pipeline code into reusable functions or classes
- Add config support to easily swap models, tokenizers, data formats (including multilabel)
- Add SBERT or TF-IDF + classifier as baselines
- Explore data augmentation techniques, especially for underrepresented classes like "dramatic"
- Export EDA and model evaluation to interactive reports
- Add user-friendly web UI (e.g., via Streamlit)
- Prepare support for multilabel genre classification as a bonus extension

Author
------
Ana Magalhães
