import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_data(dataset_path, img_size=(30, 30)):
    """
    Loads images and labels from the GTSRB Train dataset.
    
    Args:
        dataset_path (str): Path to the folder containing 'Train/' folder and 'Train.csv'
        img_size (tuple): Target image size.
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    images = []
    labels = []
    
    train_dir = os.path.join(dataset_path, 'Train')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory 'Train' not found in {dataset_path}")
    
    # Iterate through folders 0 - 42
    for class_id in range(43):
        path = os.path.join(train_dir, str(class_id))
        if not os.path.exists(path):
            continue
            
        for img_file in os.listdir(path):
            try:
                img_path = os.path.join(path, img_file)
                image = cv2.imread(img_path)
                image = cv2.resize(image, img_size)
                images.append(image)
                labels.append(class_id)
            except Exception as e:
                print(f"Error loading image {img_file}: {e}")
                
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize images
    images = images / 255.0
    
    # Split into Train/Val
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=43)
    y_val = to_categorical(y_val, num_classes=43)
    
    return X_train, X_val, y_train, y_val

def load_test_data(dataset_path, img_size=(30, 30)):
    """
    Loads test images and labels from 'Test/' and 'Test.csv'.
    """
    test_csv_path = os.path.join(dataset_path, 'Test.csv')
    test_dir = os.path.join(dataset_path, 'Test')
    
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test.csv not found in {dataset_path}")
        
    df = pd.read_csv(test_csv_path)
    labels = df['ClassId'].values
    img_paths = df['Path'].values
    
    images = []
    for path in img_paths:
        try:
            full_path = os.path.join(dataset_path, path)
            image = cv2.imread(full_path)
            image = cv2.resize(image, img_size)
            images.append(image)
        except Exception as e:
            print(f"Error loading test image {path}: {e}")
            
    images = np.array(images) / 255.0
    labels = to_categorical(labels, num_classes=43)
    
    return images, labels
