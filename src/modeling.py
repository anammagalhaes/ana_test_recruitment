# modeling.py
# Treinamento do modelo com suporte a checkpoint, métricas e salvamento de previsões

import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, ClassLabel
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }


def run_model_training(
    train_path: str = "data/train_data.csv",
    test_path: str = "data/test_data.csv",
    output_dir: str = "results",
    model_output_dir: str = "modelo_distilbert",
    num_epochs: int = 1,
    max_steps: Optional[int] = None,
    batch_size_train: int = 8,
    batch_size_eval: int = 8,
    learning_rate: float = 2e-5
) -> None:
    print("[INFO] Loading training and test datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"[INFO] Train samples: {len(df_train)} | Test samples: {len(df_test)}")

    print("[INFO] Encoding labels...")
    label_encoder = LabelEncoder()
    df_train["label"] = label_encoder.fit_transform(df_train["target"])
    df_test["label"] = label_encoder.transform(df_test["target"])
    labels = label_encoder.classes_.tolist()

    print("[INFO] Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["synopsis"], padding=True, truncation=True)

    print("[INFO] Converting to Hugging Face Datasets...")
    train_dataset = Dataset.from_pandas(df_train[["synopsis", "label"]])
    test_dataset = Dataset.from_pandas(df_test[["synopsis", "label"]])
    train_dataset = train_dataset.cast_column("label", ClassLabel(num_classes=len(labels), names=labels))
    test_dataset = test_dataset.cast_column("label", ClassLabel(num_classes=len(labels), names=labels))

    print("[INFO] Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    print("[INFO] Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(labels)
    )

    print("[INFO] Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_eval,
        num_train_epochs=1 if max_steps else num_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro"
    )

    print("[INFO] Initializing Hugging Face Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Suporte a checkpoint
    resume_path: Optional[str] = None
    if os.path.isdir(output_dir):
        checkpoints = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint")
        ]
        if checkpoints:
            checkpoints.sort()
            resume_path = checkpoints[-1]
            print(f"[INFO] Resuming from checkpoint: {resume_path}")

    print("[INFO] Starting training...")
    trainer.train(resume_from_checkpoint=resume_path)

    print("[INFO] Saving model and tokenizer...")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    # === Salvar previsões no conjunto de teste ===
    print("[INFO] Saving test set predictions...")
    raw_preds = trainer.predict(test_dataset)
    pred_labels = torch.argmax(torch.tensor(raw_preds.predictions), dim=1).numpy()

    df_test["predicted"] = label_encoder.inverse_transform(pred_labels)
    df_test["true"] = label_encoder.inverse_transform(df_test["label"])

    os.makedirs(output_dir, exist_ok=True)
    df_test.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    print("[SUCCESS] Training complete, predictions saved.")

if __name__ == "__main__":
    run_model_training()
