from tensorflow import keras 
from keras import layers, models, metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Enable GPU memory growth (prevents OOM errors)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU available: {len(gpus)} GPU(s) detected")
    except RuntimeError as e:
        print(e)
else:
    print("⚠ No GPU detected - training on CPU")

# Enable mixed precision for faster training on GPU
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load and prepare the data
df = pd.read_csv('/content/drive/MyDrive/Colab/fer2013.csv')
X = np.array([np.fromstring(x, sep=' ') for x in df['pixels'].values], dtype=np.float32)
X = X.reshape(-1, 48, 48, 1)  # Reshape to (samples, height, width, channels)
X = X / 255.0  # Normalize pixel values

# Convert labels to one-hot encoding
y = pd.get_dummies(df['emotion']).values.astype(np.float32)
labels = np.argmax(y, axis=1)  # For proper stratification

# Split the data into training (70%) and test (30%) sets
X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
    X, y, labels,
    test_size=0.1,
    random_state=42,
    stratify=labels 
)

# Further split training data into training (70%) and validation (16% of total)
X_train, X_val, y_train, y_val, labels_train, labels_val = train_test_split(
    X_train, y_train, labels_train,
    test_size=0.1,
    random_state=42,
    stratify=labels_train
)

# Data augmentation layers (applied only during training)
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),  # 10% rotation
    layers.RandomZoom(0.1),  # 10% zoom
], name="data_augmentation")

num_classes = 7

model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),
    data_augmentation,
    
    # Block 1
    layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Block 3
    layers.Conv2D(256, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(256, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    
    # Output layer (float32 for numerical stability with mixed precision)
    layers.Dense(num_classes, activation='softmax', dtype='float32')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', metrics.F1Score(name='f1_score', average='macro')]
)

# Compute class weights to handle class imbalance
class_counts = np.bincount(labels_train, minlength=num_classes)
total = class_counts.sum()
class_weight = {i: float(total / (num_classes * class_counts[i])) if class_counts[i] > 0 else 0.0 for i in range(num_classes)}

print(f"Class distribution: {class_counts}")
print(f"Class weights: {class_weight}")

# Callbacks for better training
callbacks = [
    EarlyStopping(monitor='val_f1_score', mode='max', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_f1_score', mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_emotion_model.keras', monitor='val_f1_score', mode='max', save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    batch_size=128,  # Increased from 32 for better GPU utilization
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

model.save('emotion_model.keras')

# Print final metrics
best_val_f1 = max(history.history.get('val_f1_score', [0.0]))
best_val_acc = max(history.history.get('val_accuracy', [0.0]))
print(f"\n{'='*50}")
print(f"Best Validation F1 Score: {best_val_f1:.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")
print(f"{'='*50}")

# Evaluate on test set
test_results = model.evaluate(X_test, y_test, return_dict=True, verbose=0)
print(f"Test Accuracy: {test_results['accuracy']:.4f}")
print(f"Test F1 Score: {test_results['f1_score']:.4f}")
print(f"{'='*50}")