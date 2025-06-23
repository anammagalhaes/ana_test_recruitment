from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

from src.utils import clean_text

app = FastAPI(title="NLP Movie Classifier API")

# ------ Load model and tokenizer ------
try:
    model = DistilBertForSequenceClassification.from_pretrained("modelo_distilbert")
    tokenizer = DistilBertTokenizerFast.from_pretrained("modelo_distilbert")
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading/tokenizer {e}")

labels = ["cult", "dramatic", "paranormal"]

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="O campo 'text' está vazio.")

    # ------ Clean input text ------
    cleaned_text = clean_text(input.text)

    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    return {"prediction": labels[pred]}

@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):
    try:
        content = file.file.read().decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="Erro ao ler arquivo enviado.")
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="Arquivo vazio.")

    # ------ Clean file content ------
    cleaned_text = clean_text(content)

    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()

    return {"prediction": labels[pred]}
