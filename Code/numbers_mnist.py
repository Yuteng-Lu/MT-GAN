# coding=utf-8

import tensorflow as tf
from tensorflow import keras
import network

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
train_images, test_images = train_images / 255., test_images / 255.


def create_model(new=False):
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            test()

    def test():
        model.evaluate(test_images, test_labels)
        print()

    if new:
        callbacks = MyCallback()
        model = network.CNNNetwork().create_CNN_model_1()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.summary()
        model.fit(train_images, train_labels, epochs=5, callbacks=callbacks)
    else:
        try:
            model = keras.models.load_model('CNN.h5')
            # model.summary()
            # test()
        except OSError:
            callbacks = MyCallback()
            model = keras.Sequential([
                keras.layers.Conv2D(64, (3, 3), activation='relu',
                                    input_shape=(28, 28, 1)),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(10, activation='softmax'),
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
            model.summary()
            model.fit(train_images, train_labels, epochs=5, callbacks=callbacks)
            model.save('CNN.h5')
    return model


if __name__ == '__main__':
    my_model = create_model()
