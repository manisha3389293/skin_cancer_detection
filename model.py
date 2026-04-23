import numpy as np [cite: 30]
import pandas as pd [cite: 31]
import matplotlib.pyplot as plt [cite: 33]
from glob import glob [cite: 34]
import tensorflow as tf [cite: 37]
from tensorflow.keras import layers, Model [cite: 39, 203]
from tensorflow.keras.applications.efficientnet import EfficientNetB7 [cite: 192]

# 1. Load Data Paths [cite: 44]
images = glob('train_cancer/*/*.jpg') [cite: 46]
df = pd.DataFrame({'filepath': images}) [cite: 54]
df['label'] = df['filepath'].str.split('/', expand=True)[1] [cite: 55]
df['label_bin'] = np.where(df['label'].values == 'malignant', 1, 0) [cite: 94]

# 2. Preprocessing Function [cite: 150]
def decode_image(filepath, label=None):
    img = tf.io.read_file(filepath) [cite: 153]
    img = tf.image.decode_jpeg(img) [cite: 154]
    img = tf.image.resize(img, [224, 224]) [cite: 155]
    img = tf.cast(img, tf.float32) / 255.0 [cite: 156]
    return img, label [cite: 162]

# 3. Model Architecture (Transfer Learning) [cite: 179, 183]
base_model = EfficientNetB7(input_shape=(224, 224, 3), weights='imagenet', include_top=False) [cite: 194, 195, 197]
for layer in base_model.layers:
    layer.trainable = False [cite: 198, 199]

x = layers.Flatten()(base_model.output) [cite: 187, 206]
x = layers.Dense(256, activation='relu')(x) [cite: 208]
x = layers.BatchNormalization()(x) [cite: 209]
x = layers.Dropout(0.3)(x) [cite: 214]
outputs = layers.Dense(1, activation='sigmoid')(x) [cite: 217, 220]

model = Model(base_model.input, outputs) [cite: 218]

# 4. Compile and Train [cite: 219, 229]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC']) [cite: 225, 226, 227, 228]
# model.fit(train_ds, validation_data=val_ds, epochs=5)