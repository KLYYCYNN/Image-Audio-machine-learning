import numpy as np
from tensorflow.keras import layers, models, optimizers, losses, Input, activations, preprocessing, applications, metrics, utils, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt

# %%

train_ds = utils.image_dataset_from_directory(
    "data/Training",              # path to folder
    image_size=(300, 300),        # resize all images to this size
    batch_size=32,                # images per batch
    label_mode="int",             # "int", "categorical", or "binary"
    shuffle=True,                 
    seed=123                      # for reproducibility
)

# Load validation dataset
val_ds = utils.image_dataset_from_directory(
    "data/Dev",
    image_size=(300,300),
    batch_size=32,
    label_mode="int"
)

# Load test dataset (assuming you created a "Test" folder)
test_ds = utils.image_dataset_from_directory(
    "data/Testing",
    image_size=(300,300),
    batch_size=10,
    label_mode="int",
    shuffle=False # No need to shuffle the test set
)

# Apply the same preprocessing



def preprocess_image(image, label):
    return applications.efficientnet.preprocess_input(image), label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocess_image).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(preprocess_image).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(preprocess_image).cache().prefetch(buffer_size=AUTOTUNE)
# %%


base_model = applications.EfficientNetB3(include_top=False, input_shape=(300, 300, 3), weights='imagenet')
base_model.trainable = False  # Freeze base model initially

inputs = Input(shape=(300, 300, 3))

# 2. Apply augmentation layers.
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.1)(x)
x = layers.RandomContrast(0.2)(x)

# NOTE: We do not add the applications.efficientnet.preprocess_input here
# because you have already applied it to the dataset using the .map() method.

# 3. Pass the augmented data to the base model in inference mode.
x = base_model(x, training=False)

# 4. Add your custom classification head.
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(units=3, activation='softmax')(x)

# 5. Create the final model, ensuring the 'inputs' variable is the same
#    one from step 1.
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# You can optionally print the summary to verify the architecture
model.summary()

#%%
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=15, # Wait for 15 epochs without improvement
    restore_best_weights=True,
    verbose=1
)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2, 
    patience=5, # Wait for 5 epochs without improvement
    min_lr=1e-7, # Minimum learning rate
    verbose=1
)

# --- Stage 1: Freeze base_model and train head ---
base_model.trainable = False
model.compile(
    optimizer = optimizers.Adam(learning_rate = 1e-4), # Initial learning rate for head
    loss = losses.SparseCategoricalCrossentropy(),
    metrics = [metrics.SparseCategoricalAccuracy()]
)

#%%

print("--- Starting Stage 1: Training Head (Base Model Frozen) ---")
history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50, 
    verbose=1,
    callbacks=[early_stopping_callback, reduce_lr_callback] 
)

#%%

# --- Stage 2: Unfreeze base_model and fine-tune ---
base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = False

# Then, unfreeze the top N layers
# You can experiment with this number
layers_to_unfreeze = 40 
for layer in base_model.layers[-layers_to_unfreeze:]:
    layer.trainable = True

model.compile(
    optimizer = optimizers.Adam(learning_rate = 1e-5), 
    loss=losses.SparseCategoricalCrossentropy(),
    metrics = [metrics.SparseCategoricalAccuracy()]
)

#%%

print("\n--- Starting Stage 2: Fine-tuning (Base Model Unfrozen) ---")

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=200, # Set a much higher max epoch, early stopping will handle it
    verbose=1,
    callbacks=[early_stopping_callback, reduce_lr_callback] # Use both callbacks
)

# %%
import matplotlib.pyplot as plt



plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_stage2.history['loss'], label='Training Loss')
plt.plot(history_stage2.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_stage2.history['sparse_categorical_accuracy'], label='Training Accuracy')
plt.plot(history_stage2.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()
# %%

print("\n--- Evaluating Final Model on Test Data ---")
final_loss, final_accuracy = model.evaluate(test_ds)
print(f"Final Test Accuracy: {final_accuracy*100:.2f}%")
