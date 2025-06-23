from src import preprocessing, modeling, evaluate_model
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument("--fast", action="store_true", help="Executar em modo rápido (debug)")
    args = parser.parse_args()

    print("=== Etapa 1: Data Processing ===")
    preprocessing.run_preprocessing(fast=args.fast)

    print("=== Etapa 2: Model Training ===")
    if args.fast:
        modeling.run_model_training(
            num_epochs=4,
            batch_size_train=8,
            batch_size_eval=8,
            learning_rate=2e-5,
            max_steps=None
        )
    else:
        modeling.run_model_training()

    print("=== Etapa 3: Model Evaluation ===")
    evaluate_model.run_evaluation()

    print(" === Pipeline Executed ===")

if __name__ == "__main__":
    main()
