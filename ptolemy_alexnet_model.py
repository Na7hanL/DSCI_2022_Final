import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import json
import time

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.client import device_lib


if __name__ == '__main__':
    # Name used for saving files
    run_name = 'ptolemy_alex_net'

    print(device_lib.list_local_devices())
    
    data_dir = './images'
    data_dir = pathlib.Path(data_dir).with_suffix('')

    start = time.time()

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"\nTotal Number of Images: {image_count}")

    batch_size = 64
    img_height = 200
    img_width = 200

    print('\nCollecting Training Data...')
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    print('\nCollecting Validation Data...')
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    # Early stopping callback function that stops training when validation loss has not improved for 3 consequetive epochs
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model = Sequential([
      layers.RandomFlip("horizontal", input_shape=(img_height,img_width,3)),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.1),
      layers.Rescaling(1./255),
      keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu'),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2)),
      keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(3,3)),
      keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
      keras.layers.BatchNormalization(),
      keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
      keras.layers.BatchNormalization(),
      keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(2,2)),
      keras.layers.Flatten(),
      keras.layers.Dense(1024,activation='relu'),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(512,activation='relu'),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.summary()

    epochs=50
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      callbacks=[callback]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    history.history['training_time'] = time.time() - start

    with open(run_name + '.json', 'w') as outfile:
        json.dump(history.history, outfile)

    print(f"Validation Accuracy: {val_acc}")

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.ylim(0,1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(run_name + ".png")
    plt.show()

