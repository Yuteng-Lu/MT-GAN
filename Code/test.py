from random import choice

from tensorflow.python.keras.models import load_model
from mnist_GAN_full_dataset_batchup import load_real_samples, summarize_performance
from numpy import concatenate, zeros
from numpy.random import randn, seed
from matplotlib import pyplot
# from mnist_GAN_full_dataset_exp import noise_exp
from mnist_GAN_full_dataset_Gamma import noise_Gamma
# from mnist_GAN_full_dataset_Gauss import noise_Gauss
# from keras.datasets.mnist import load_data
from numbers_mnist import create_model


def play1():
    # size of the latent space
    latent_dim = 100
    # load image data
    dataset = load_real_samples()
    # standard d model
    standard_d_model = load_model('data/full/discriminator_model_100.h5')
    # manually enumerate epochs
    for i in range(10):
        # evaluate the model performance, sometimes
        n = 10 * (i + 1)
        g_model = load_model('data/gamma_pp/Gamma_generator_model_%03d.h5' % n)
        summarize_performance(i, g_model, standard_d_model, dataset, latent_dim, save=False)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    seed(158)
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = g_model.predict(x_input)
    return x


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


def play2():
    # size of the latent space
    latent_dim = 100
    # models = [
    #     'data/full/generator_model_100.h5',
    #     'data/batchup/bu_generator_model_100.h5',
    #     'data/batchdown/bd_generator_model_100.h5',
    #     'data/exp/exp_generator_model_100.h5',
    #     'data/gamma/Gamma_generator_model_100.h5',
    #     'data/gauss/Gauss_generator_model_100.h5',
    #     'data/gamma_partial/Gamma_generator_model_100.h5',
    #     'data/gauss_partial/Gauss_generator_model_100.h5',
    #     'data/rr37/rr_generator_model_100.h5',
    #     'data/cs/cs_generator_model_100.h5'
    # ]
    models = [
        'data/cd_minus1/cd_generator_model_100.h5',
        'data/cd0/cd_generator_model_100.h5',
        'data/cd0_2/cd_generator_model_100.h5',
        'data/cd2/cd_generator_model_100.h5',
        'data/cd2_2/cd_generator_model_100.h5',
        'data/rr/rr_generator_model_100.h5',
        'data/exp_pp/exp_generator_model_100.h5',
        'data/gamma_pp/Gamma_generator_model_100.h5',
        'data/gauss_pp/Gauss_generator_model_100.h5',
        'data/add_copy/add_generator_model_100.h5'
    ]
    x_fake = []
    for model in models:
        # load g_model
        g_model = load_model(model)
        # custom_objects={'leaky_relu': tf.nn.leaky_relu}
        # generate images
        images = generate_fake_samples(g_model, latent_dim, 10)
        x_fake.append(images)
    x_fake = concatenate(x_fake)
    save_plot(x_fake, 99)
    model = create_model()
    my_list = list(map(argmax, model.predict(x_fake)))
    for i in range(100):
        print(my_list[i], end='')
        if i % 10 == 9:
            print()
        else:
            print(' ', end='')


def play3():
    image = zeros((28, 28))
    # (trainX, _), (_, _) = load_data()
    # image = trainX[0]
    image = noise_Gamma(image)
    pyplot.axis('off')
    pyplot.imshow(image, cmap='gray_r')
    filename = 'generated_plot_e000.png'
    pyplot.savefig(filename)
    pyplot.close()


def argmax(s):
    max_index_list = []
    max_value = s[0]
    for index, value in enumerate(s):
        if value > max_value:
            max_index_list.clear()
            max_value = value
            max_index_list.append(index)
        elif value == max_value:
            max_index_list.append(index)
    return choice(max_index_list)


def play4():
    # size of the latent space
    latent_dim = 100
    # load g_model
    g_model = load_model('data/gamma_pp/Gamma_generator_model_100.h5')
    # generate images
    x_fake = generate_fake_samples(g_model, latent_dim, 100)
    save_plot(x_fake, 99)
    model = create_model()
    my_list = list(map(argmax, model.predict(x_fake)))
    for i in range(100):
        print(my_list[i], end='')
        if i % 10 == 9:
            print()
        else:
            print(' ', end='')


play1()
