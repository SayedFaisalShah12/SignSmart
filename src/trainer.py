import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from src.model_factory import build_custom_cnn, build_pretrained_mobilenet

def train_model(X_train, X_val, y_train, y_val, model_type='custom', epochs=30, batch_size=32):
    """
    Trains a model with data augmentation.
    
    Args:
        X_train, X_val, y_train, y_val: Training and validation data.
        model_type: 'custom' or 'pretrained'.
        epochs: Number of epochs to train.
        batch_size: Training batch size.
        
    Returns:
        tuple: (trained_model, training_history)
    """
    # Load architecture
    if model_type == 'custom':
        model = build_custom_cnn(input_shape=X_train.shape[1:])
    else:
        # Transfer Learning requires images resized to 224x224
        model = build_pretrained_mobilenet(input_shape=X_train.shape[1:])
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Data Augmentation (Essential for small datasets)
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,  # Signs are usually direction-sensitive
        vertical_flip=False,
        fill_mode="nearest")

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"--- Training {model_type} Model ---")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stop]
    )
    
    return model, history

def save_model(model, model_name, save_dir='models'):
    """
    Saves the trained model to the specified directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f"{model_name}.h5")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return save_path
