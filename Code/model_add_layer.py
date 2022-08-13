import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from numpy.random import randint


def add_mut_layer(name):
    model = keras.models.load_model(f'{name}.h5')
    new_model = Sequential()
    for i in range(2):
        new_model.add(model.layers[i])
    new_model.add(mut_layer())
    for i in range(2, len(model.layers)):
        new_model.add(model.layers[i])
    new_model.save(f'add_after_{name}.h5')
    return new_model


def mut_layer():
    n_nodes = 128 * 7 * 7
    model = Sequential()
    model.add(Dense(n_nodes, input_dim=n_nodes))
    model.add(LeakyReLU(alpha=0.2))
    my_list = [randint(0, 2) for _ in range(n_nodes)]
    model.layers[0].weights[0].assign([my_list for _ in range(n_nodes)])
    # print(model.layers)
    return model


add_mut_layer('generator_model_070')
mut_layer()
