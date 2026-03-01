import argparse
import os
import tensorflow as tf
from src.data_loader import load_data, load_test_data
from src.trainer import train_model, save_model
from src.evaluator import plot_training_history, evaluate_model

# Constants
DATASET_PATH = 'dataset'
IMG_SIZE_CUSTOM = (30, 30)
IMG_SIZE_PRETRAINED = (224, 224)

def main(args):
    """
    Main entry point for training and evaluation.
    """
    # Decide image size based on model type
    img_size = IMG_SIZE_CUSTOM if args.model_type == 'custom' else IMG_SIZE_PRETRAINED
    
    # Check if dataset directory exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset directory '{DATASET_PATH}' not found.")
        print("Please download GTSRB from Kaggle and place it in the 'dataset' folder.")
        print("Structure should be: dataset/Train/... and dataset/Test.csv")
        return

    # 1. Load Data
    print(f"Loading data with image size {img_size}...")
    X_train, X_val, y_train, y_val = load_data(DATASET_PATH, img_size=img_size)
    print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    
    # 2. Train Model
    model, history = train_model(X_train, X_val, y_train, y_val, 
                                 model_type=args.model_type, 
                                 epochs=args.epochs, 
                                 batch_size=args.batch_size)
    
    # 3. Save Model
    save_model(model, args.model_type)
    
    # 4. Plot History
    plot_training_history(history)
    
    # 5. Evaluate on Test Set
    print("\n--- Testing Model ---")
    try:
        X_test, y_test = load_test_data(DATASET_PATH, img_size=img_size)
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        print(f"Skipping final test evaluation: {e}")
        print("Final validation accuracy was: ", history.history['val_accuracy'][-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignSmart: Traffic Signal Recognition")
    parser.add_argument("--model-type", type=str, default="custom", choices=["custom", "pretrained"],
                        help="Choose model architecture ('custom' or 'pretrained')")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    main(args)
