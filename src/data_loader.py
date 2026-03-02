import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(dataset_path, img_size=(30, 30), batch_size=32):
    """
    Creates memory-efficient data generators for training and validation.
    
    Args:
        dataset_path (str): Path to the folder containing 'Train' folder.
        img_size (tuple): Target image size.
        batch_size (int): Batch size.
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    train_dir = os.path.join(dataset_path, 'Train')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory 'Train' not found in {dataset_path}")

    # Data Augmentation & Normalization for Training
    train_datagen = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        validation_split=0.2  # 20% for validation
    )

    # Training Generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation Generator
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator

def get_test_generator(dataset_path, img_size=(30, 30), batch_size=32):
    """
    Creates a memory-efficient generator for the test dataset using Test.csv.
    """
    test_csv_path = os.path.join(dataset_path, 'Test.csv')
    
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test.csv not found in {dataset_path}")
        
    df = pd.read_csv(test_csv_path)
    # Ensure ClassId is string for categorical mode
    df['ClassId'] = df['ClassId'].astype(str)
    
    test_datagen = ImageDataGenerator(rescale=1/255.0)
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        directory=dataset_path,
        x_col="Path",
        y_col="ClassId",
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

