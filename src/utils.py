# utils.py

import re
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'[;=]{3,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_and_clean_data(raw_path: str) -> pd.DataFrame:
    """
    Lê o arquivo original, reconstrói colunas, limpa e normaliza o dataset.
    """
    pattern = r'["“]?\s*(\d+)\s*,\s*(.*?)\s*,\s*["“](.*?)["”]\s*,\s*(cult|paranormal|dramatic)["”]?;'
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    df = pd.DataFrame(matches, columns=["id", "title", "synopsis", "target"])

    df["synopsis"] = df["synopsis"].apply(clean_text).str.lower()
    df["title"] = df["title"].apply(clean_text)
    df["target"] = df["target"].apply(clean_text).str.lower()

    df = df.drop_duplicates(subset="synopsis").reset_index(drop=True)
    return df
