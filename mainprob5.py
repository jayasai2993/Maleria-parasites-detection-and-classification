#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import numpy as np
import matplotlib.pyplot as plt



# Set the dataset directory
dataset_dir = r'C:\Users\SAI\Downloads\Malaria-20241008T052442Z-002\Malaria'

# Check if the directory exists
if not os.path.exists(dataset_dir):
    print("Dataset directory does not exist.")
else:
    print("Dataset directory exists.")

# Get the number of classes from the subdirectories in your dataset
# Assuming two main directories: "Infected" and "Not Infected"
main_classes = ['Infected', 'Not Infected']
num_main_classes = len(main_classes)

# Load pre-trained Xception without the top classification layer for main classification
base_model_main = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new classification layers for main classification
x = base_model_main.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # Dense layer with ReLU
predictions_main = Dense(num_main_classes, activation='softmax')(x)  # Final layer for main classification

# Create the main model
model_main = Model(inputs=base_model_main.input, outputs=predictions_main)

# Freeze the base layers for transfer learning
for layer in base_model_main.layers:
    layer.trainable = False

# Compile the main model
model_main.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create an image data generator for data augmentation for main classification
train_datagen_main = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2
)

# Load your dataset for main classification
train_generator_main = train_datagen_main.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator_main = train_datagen_main.flow_from_directory(
    dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Callbacks for main classification
early_stopping_main = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint_main = ModelCheckpoint('best_model_xception_main.keras', save_best_only=True, monitor='val_loss')
reduce_lr_main = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the main model
history_main = model_main.fit(
    train_generator_main,
    validation_data=validation_generator_main,
    epochs=10,
    callbacks=[early_stopping_main, model_checkpoint_main, reduce_lr_main]
)

# Now create a model for infected stage classification
# Get the number of stages for infected images (4 stages)
infected_dir = os.path.join(dataset_dir, 'infected')
num_stage_classes = len(os.listdir(infected_dir))  # Number of subdirectories in 'Infected'

# Load pre-trained Xception without the top classification layer for stage classification
base_model_stage = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new classification layers for stage classification
x_stage = base_model_stage.output
x_stage = GlobalAveragePooling2D()(x_stage)
x_stage = Dense(128, activation='relu')(x_stage)  # Dense layer with ReLU
predictions_stage = Dense(num_stage_classes, activation='softmax')(x_stage)  # Final layer for stage classification

# Create the stage model
model_stage = Model(inputs=base_model_stage.input, outputs=predictions_stage)

# Freeze the base layers for transfer learning
for layer in base_model_stage.layers:
    layer.trainable = False

# Compile the stage model
model_stage.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create an image data generator for data augmentation for stage classification
train_datagen_stage = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2
)

# Load your dataset for stage classification
train_generator_stage = train_datagen_stage.flow_from_directory(
    infected_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator_stage = train_datagen_stage.flow_from_directory(
    infected_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Callbacks for stage classification
early_stopping_stage = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint_stage = ModelCheckpoint('best_model_xception_stage.keras', save_best_only=True, monitor='val_loss')
reduce_lr_stage = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the stage model
history_stage = model_stage.fit(
    train_generator_stage,
    validation_data=validation_generator_stage,
    epochs=10,
    callbacks=[early_stopping_stage, model_checkpoint_stage, reduce_lr_stage]
)

# Evaluate the models
loss_main, accuracy_main = model_main.evaluate(validation_generator_main)
print(f"Main Validation Loss: {loss_main}, Main Validation Accuracy: {accuracy_main}")


# In[2]:


loss_stage, accuracy_stage = model_stage.evaluate(validation_generator_stage)
print(f"Stage Validation Loss: {loss_stage}, Stage Validation Accuracy: {accuracy_stage}")

# Visualization of predictions for main classification
plt.figure(figsize=(20, 20))
class_names_main = train_generator_main.class_indices  # Get main class indices
class_names_main = {v: k for k, v in class_names_main.items()}  # Reverse the dictionary

# Take one batch of validation data for main classification
images_main, labels_main = next(validation_generator_main)  # Get one batch

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images_main[i])  # No need to convert to uint8 as they are already scaled
    predictions_main = model_main.predict(tf.expand_dims(images_main[i], axis=0))

    score_main = tf.nn.softmax(predictions_main[0])
    predicted_class_main = class_names_main[np.argmax(score_main)]
    actual_class_main = class_names_main[np.argmax(labels_main[i])]

    plt.title(f"Actual: {actual_class_main}")

    if actual_class_main == predicted_class_main:
        plt.ylabel(f"Predicted: {predicted_class_main}", fontdict={'color': 'green'})
    else:
        plt.ylabel(f"Predicted: {predicted_class_main}", fontdict={'color': 'red'})

    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])

plt.tight_layout()
plt.show()


# In[3]:


# Visualization of predictions for stage classification
plt.figure(figsize=(20, 20))
class_names_stage = train_generator_stage.class_indices  # Get stage class indices
class_names_stage = {v: k for k, v in class_names_stage.items()}  # Reverse the dictionary

# Take one batch of validation data for stage classification
images_stage, labels_stage = next(validation_generator_stage)  # Get one batch

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images_stage[i])  # No need to convert to uint8 as they are already scaled
    predictions_stage = model_stage.predict(tf.expand_dims(images_stage[i], axis=0))

    score_stage = tf.nn.softmax(predictions_stage[0])
    predicted_class_stage = class_names_stage[np.argmax(score_stage)]
    actual_class_stage = class_names_stage[np.argmax(labels_stage[i])]

    plt.title(f"Actual: {actual_class_stage}")

    if actual_class_stage == predicted_class_stage:
        plt.ylabel(f"Predicted: {predicted_class_stage}", fontdict={'color': 'green'})
    else:
        plt.ylabel(f"Predicted: {predicted_class_stage}", fontdict={'color': 'red'})

    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels


# In[4]:


# After training the main model
model_main.save('best_model_xception_main.keras')  # Save the entire model (architecture + weights)

# After training the stage model
model_stage.save('best_model_xception_stage.keras')  # Save the entire model (architecture + weights)


# In[ ]:




