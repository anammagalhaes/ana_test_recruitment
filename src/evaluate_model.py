import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

def run_evaluation(
    test_path="data/test_data.csv",
    model_dir="modelo_distilbert",
    batch_size=8
):
    print("[INFO] Carregando modelo e tokenizer...")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model.eval()

    print("[INFO] Carregando dados de teste...")
    df = pd.read_csv(test_path)
    labels = sorted(df["target"].unique())
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["target"])

    print("[INFO] Tokenizando os dados...")
    dataset = Dataset.from_pandas(df[["synopsis", "label"]])
    dataset = dataset.map(lambda x: tokenizer(x["synopsis"], truncation=True, padding=True), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print("[INFO] Realizando predições...")
    preds, true_labels = [], []
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits
            pred = torch.argmax(logits, axis=1)
            preds.extend(pred.cpu().numpy())
            true_labels.extend(batch["label"].cpu().numpy())

    print("\n=== Classification Report ===")
    report = classification_report(true_labels, preds, target_names=label_encoder.classes_, digits=4)
    print(report)

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(true_labels, preds))

    history_path = "training_history.csv"
    if os.path.exists(history_path):
        print("[INFO] Plotando histórico de treino...")    #TODO
        history = pd.read_csv(history_path)

        if "loss" in history.columns and "eval_loss" in history.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(history["epoch"], history["loss"], label="Train Loss")
            plt.plot(history["epoch"], history["eval_loss"], label="Eval Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Train vs Eval Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("loss_plot_eval.png")
            plt.close()
            print("[INFO] Gráfico salvo como loss_plot_eval.png")   #TODO
        else:
            print("[WARNING] training_history.csv não contém colunas esperadas.")   #TODO
    else:
        print("[INFO] training_history.csv não encontrado — gráfico não gerado.")   #TODO

if __name__ == "__main__":
    run_evaluation()
