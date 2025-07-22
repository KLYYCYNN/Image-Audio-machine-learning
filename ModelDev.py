import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses, Input, activations, preprocessing, applications, metrics

# %%

base_model = applications.EfficientNetB3(include_top=False, input_shape=(300, 300, 3), weights='imagenet')
base_model.trainable = False  # Freeze base model initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(units=3, activation='linear')
])

model.compile( optimizer = optimizers.Adam(learning_rate = 1e-3),
               loss = losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics = [metrics.SparseCategoricalAccuracy()]
              )
