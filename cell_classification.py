from keras.layers import *
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
from keras.regularizers import l1
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator


# data path
train_path = "../DATASET/cell_images/train"
test_path = "../DATASET/cell_images/test"

# generators
train_generator = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                     height_shift_range=0.2, shear_range=0.2, zoom_range=0.3,
                                     fill_mode="nearest")
test_generator = ImageDataGenerator(rescale=1./255)

# flow from directory
train_set = train_generator.flow_from_directory(directory=train_path, shuffle=True,
                                                target_size=(28, 28), class_mode="categorical", color_mode="grayscale")

test_set = test_generator.flow_from_directory(directory=test_path, shuffle=True, target_size=(28, 28),
                                              class_mode="categorical", color_mode="grayscale")

# callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", start_from_epoch=3)
tensorboard  = TensorBoard(log_dir="../tensor_boards/cell", histogram_freq=1,
                           write_images=True, write_graph=True, update_freq="epoch", profile_batch=2,
                           embeddings_freq=1)
model_checkpoint = ModelCheckpoint(filepath="../models/cell_model", save_best_only=True)
callbacks_list = [early_stopping, tensorboard, model_checkpoint]

# model
model = Sequential()

# conv1
model.add(Conv2D(filters=64, kernel_size=(4, 4), activation="relu", input_shape=(28, 28, 1), padding="valid",
                 strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# conv2
model.add(Conv2D(filters=64, kernel_size=(4, 4), activation="relu", padding="valid", strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# batchnormalization
model.add(BatchNormalization())

# flatten
model.add(Flatten())

# Dense
model.add(Dense(512, activation="relu", kernel_regularizer=l1(0.001)))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu", kernel_regularizer=l1(0.001)))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))

# summary
model.summary()
# model compile
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# fitting
model.fit(train_set, epochs=200, batch_size=128, shuffle=True, validation_split=0.2, callbacks=callbacks_list,
          verbose=1, validation_data=test_set)


