import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses, Input, activations, preprocessing, applications, metrics, utils
import tensorflow as tf
# %%

train_ds = utils.image_dataset_from_directory(
    "data/Training",              # path to folder
    image_size=(300, 300),        # resize all images to this size
    batch_size=16,                # images per batch
    label_mode="int",             # "int", "categorical", or "binary"
    shuffle=True,                 
    seed=123                      # for reproducibility
)

# Load validation dataset
val_ds = utils.image_dataset_from_directory(
    "data/Dev",
    image_size=(300,300),
    batch_size=16,
    label_mode="int"
)

# %%


base_model = applications.EfficientNetB3(include_top=False, input_shape=(300, 300, 3), weights='imagenet')
base_model.trainable = False  # Freeze base model initially

model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(units=3, activation='softmax')
])

model.compile( optimizer = optimizers.Adam(learning_rate = 1e-5),
               loss = losses.SparseCategoricalCrossentropy(),
               metrics = [metrics.SparseCategoricalAccuracy()]
              )

# %%

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %%

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    verbose=1
)
