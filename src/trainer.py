import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from src.model_factory import build_custom_cnn, build_pretrained_mobilenet

def train_model(train_gen, val_gen, model_type='custom', epochs=30):
    """
    Trains a model using provided data generators.
    
    Args:
        train_gen: Training data generator.
        val_gen: Validation data generator.
        model_type: 'custom' or 'pretrained'.
        epochs: Number of epochs to train.
        
    Returns:
        tuple: (trained_model, training_history)
    """
    # Determine input shape from generator
    input_shape = train_gen.image_shape
    num_classes = train_gen.num_classes

    # Load architecture
    if model_type == 'custom':
        model = build_custom_cnn(input_shape=input_shape, num_classes=num_classes)
    else:
        model = build_pretrained_mobilenet(input_shape=input_shape, num_classes=num_classes)
    
    # Compile
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"--- Training {model_type} Model ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
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
    
    save_path = os.path.join(save_dir, f"{model_name}.keras")
    model.save(save_path)
    print(f"Model saved to {save_path}")
    return save_path

