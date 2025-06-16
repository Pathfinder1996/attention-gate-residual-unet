import os
import pickle

import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from data_loader import train_generator, test_generator, save_result
from model import attention_gate_residual_unet
from my_metrics import dice_loss, iou_coefficient, dice_coefficient

plt.rcParams["font.family"] = "Times New Roman"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Data augmentation parameters
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode="nearest")

# Create training and validation data generators
train_data_gen = train_generator(16, "data/membrane/train", "image", "label", data_gen_args, save_to_dir=None)

val_aug_dict = {"rescale": 1./255}

val_data_gen = train_generator(16, "data/membrane/val", "val_image", "val_label", val_aug_dict,
                         image_color_mode="grayscale", mask_color_mode="grayscale",
                         image_save_prefix="val_image", mask_save_prefix="val_label",
                         save_to_dir=None, target_size=(256, 256), seed=1)

# Define the model
model = attention_gate_residual_unet()
model.summary()

# Compile the model with custom loss and metrics
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss=dice_loss, metrics=[iou_coefficient, dice_coefficient, "accuracy"])

# Define a model checkpoint to save the best model based on validation loss
model_checkpoint = ModelCheckpoint("best_model.hdf5", 
                                   monitor="val_loss", 
                                   verbose=1, 
                                   save_best_only=True)

# Train the model
history = model.fit(train_data_gen, 
                    epochs=100, 
                    steps_per_epoch=78, 
                    validation_data=val_data_gen, 
                    validation_steps=23,
                    callbacks=[model_checkpoint])

# Save the training history to a file
with open("trainHistory.txt", "wb") as file_pi:
    pickle.dump(history.history, file_pi)

# Plot training history
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(history.history["accuracy"], label="Train")
axs[0, 0].plot(history.history["val_accuracy"], label="Validation")
axs[0, 0].set_title("Accuracy")
axs[0, 0].set_xlabel("Epochs")
axs[0, 0].legend()

axs[0, 1].plot(history.history["loss"], label="Train")
axs[0, 1].plot(history.history["val_loss"], label="Validation")
axs[0, 1].set_title("Dice Loss")
axs[0, 1].set_xlabel("Epochs")
axs[0, 1].legend()

axs[1, 0].plot(history.history["dice_coefficient"], label="Train")
axs[1, 0].plot(history.history["val_dice_coefficient"], label="Validation")
axs[1, 0].set_title("Dice Coefficient")
axs[1, 0].set_xlabel("Epochs")
axs[1, 0].legend()

axs[1, 1].plot(history.history["iou_coefficient"], label="Train")
axs[1, 1].plot(history.history["val_iou_coefficient"], label="Validation")
axs[1, 1].set_title("IOU Coefficient")
axs[1, 1].set_xlabel("Epochs")
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# Test the model on new data
testGene = test_generator(r"D:\unet_ours\data\membrane\test")
model.load_weights("best_model.hdf5")
results = model.predict(testGene, 180, verbose=1)
save_result(r"D:\unet_ours\data\membrane\test", results)
