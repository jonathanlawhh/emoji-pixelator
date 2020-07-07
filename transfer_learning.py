from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot
import os

batch_size = 40
epochs = 5

IMG_HEIGHT, IMG_WIDTH = 200, 200

train_dir = "./dataset/train"
validation_dir = "./dataset/validation"

total_train = len(os.path.join(train_dir, 'emoji')) + len(os.path.join(train_dir, 'useless'))
total_val = len(os.path.join(validation_dir, 'emoji')) + len(os.path.join(validation_dir, 'useless'))

# Random image properties
train_image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           color_mode="rgb",
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              color_mode="rgb",
                                                              class_mode='binary')

res_net = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
res_net.trainable = False

# Quantize model
res_net = tfmot.quantization.keras.quantize_model(res_net)

tl_model = Sequential([
    res_net,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

tl_model.compile(optimizer=Adam(),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

tl_model.fit(train_data_gen,
             validation_steps=10,
             steps_per_epoch=total_train // batch_size,
             epochs=epochs,
             validation_data=val_data_gen,
             )

tl_model.save("emoji.h5")
