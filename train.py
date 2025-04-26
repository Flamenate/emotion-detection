from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load and prepare the data
df = pd.read_csv('fer2013.csv')
X = np.array([np.fromstring(x, sep=' ') for x in df['pixels'].values])
X = X.reshape(-1, 48, 48, 1)  # Reshape to (samples, height, width, channels)
X = X / 255.0  # Normalize pixel values

# Convert labels to one-hot encoding
y = pd.get_dummies(df['emotion']).values

# Split the data into training (70%) and test (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y 
)

# Further split training data into training (70%) and validation (16% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.3,
    random_state=42,
    stratify=y_train
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    X_train, y_train,
    epochs=25,
    validation_data=(X_val, y_val),
    batch_size=32,
)

model.save('emotion_model.h5')