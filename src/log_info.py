import os
import json
import torch
import pandas as pd
from transformers import TrainingArguments

def load_training_args(model_dir="modelo_distilbert"):
    args_path = os.path.join(model_dir, "training_args.bin")
    if os.path.exists(args_path):
        args = torch.load(args_path)
        print("\n=== Training Hyperparameters ===")
        print(json.dumps(args.to_dict(), indent=2))
    else:
        print(" === Training file not found ===")

def check_data_summary(train_path="data/train_data.csv", test_path="data/test_data.csv"):
    print("\n=== Dataset Sizes ===")
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        print(f"Train samples: {train_df.shape[0]}")
        print("Train columns:", list(train_df.columns))
    else:
        print(" === Training: file not found ===")

    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"Test samples: {test_df.shape[0]}")
        print("Test columns:", list(test_df.columns))
    else:
        print(" === Testing: file not found===")

def check_checkpoints(log_dir="results"):
    print("\n=== Checkpoints disponíveis ===")
    if os.path.exists(log_dir):
        checkpoints = [d for d in os.listdir(log_dir) if d.startswith("checkpoint")]
        if checkpoints:
            for ckpt in sorted(checkpoints):
                print(f"- {ckpt}")
        else:
            print(" === No checkpoint found ===")
    else:
        print(" === Pasta de resultados não encontrada.===")   #NÃO PRECISA!

def check_predictions(pred_path="results/predictions.csv"):
    print(" === Checking saved predictions ===")
    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        print(f"{len(df)} Predictions loaded ")
        print(df.head())
    else:
        print(" === No prediction file found ===")

if __name__ == "__main__":
    load_training_args()
    check_data_summary()
    check_checkpoints()
    check_predictions()
