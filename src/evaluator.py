import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_training_history(history, save_dir='plots'):
    """
    Plots the accuracy and loss curves from the training history.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()
    
    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    print(f"Training history plots saved to {save_dir}")

def evaluate_model(model, X_test, y_test, class_names=None, save_dir='plots'):
    """
    Evaluates the model on the test set and plots the confusion matrix.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print("--- Evaluating Model ---")
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Accuracy
    accuracy = np.mean(y_pred_classes == y_true)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Confusion matrix plot saved to {save_dir}")
    return accuracy
