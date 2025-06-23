import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.utils import clean_text, extract_and_clean_data

RAW_PATH = "data/task.csv"
TRAIN_PATH = "data/train_data.csv"
TEST_PATH = "data/test_data.csv"

def run_preprocessing(fast: bool = False):
    print(" === Iniciando pré-processamento ===")

    # Extrai e limpa
    df = extract_and_clean_data(RAW_PATH)

    # Reduz para testes se fast=True
    if fast:
        print(" === Modo rápido ativado: reduzindo dataset para teste ===")
        df = df.groupby("target").apply(
            lambda x: x.sample(n=min(20, len(x)), random_state=42)
        ).reset_index(drop=True)

    # Separa treino/teste
    print(" === Separando treino e teste ===")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)

    # Oversampling apenas no treino
    print(" === Balanceando o conjunto de treino === ")
    max_size = train_df["target"].value_counts().max()
    train_df = pd.concat([
        resample(g, replace=True, n_samples=max_size, random_state=42)
        for _, g in train_df.groupby("target")
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f" === Dados salvos em: {TRAIN_PATH} e {TEST_PATH} ===")