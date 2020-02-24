import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, Dense, Dropout, Flatten, Input, LeakyReLU, Reshape
from keras.optimizers import Adam
from keras.datasets.mnist import load_data
from sklearn.metrics import classification_report

from tqdm import tqdm

import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings('ignore')

from utils import plot_25


LATENT_DIM = 100
IMG_ROWS, IMG_COLS = 28, 28
N_EPOCHS = 100

def build_generator():
    generator = Sequential()
    # foundation for 7x7 image
    n_nodes = 128 * 7 * 7
    generator.add(Dense(n_nodes, input_dim=LATENT_DIM))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Reshape((7, 7, 128)))
    # upsample to 14x14
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    generator.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return generator


def build_discriminator():
    discriminator = Sequential()
    discriminator.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(IMG_ROWS, IMG_COLS, 1)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.4))
    discriminator.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.4))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(2 * 1e-4, beta_1=0.5))
    return discriminator


def build_GAN(generator, discriminator):
    discriminator.trainable = False
    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(2 * 1e-4, beta_1=0.5))
    return gan


def gen_latent(n_samples):
    v = np.random.randn(LATENT_DIM * n_samples).reshape(n_samples, LATENT_DIM)
    return v


def train_GAN(generator, discriminator, gan, X, half_batch_size=128):
    n_batch = int(len(X) / half_batch_size / 2)
    for i in tqdm(range(N_EPOCHS)):
        i += 1
        for batch in tqdm(range(n_batch)):
            noise = gen_latent(half_batch_size)
            real_idx = np.random.choice(len(X), half_batch_size)
            X_real = X[real_idx]
            X_fake = generator.predict(noise)
            X_full = np.concatenate([X_real, X_fake])
            
            # train discriminator
            discriminator.trainable = True
            y = np.zeros(half_batch_size * 2)
            y[:half_batch_size] = 1
            discriminator.train_on_batch(X_full, y)

            # train gan
            discriminator.trainable = False
            x_gan = gen_latent(half_batch_size * 2)
            y = np.ones(half_batch_size * 2)
            gan.train_on_batch(x_gan, y)
        if i % 10 == 0:
            test_noise = gen_latent(25)
            generator.save(f'models/gen_{i}.h5')
            X = generator.predict(test_noise)
            plot_25(X, i)


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = load_data()
    X_train = X_train.reshape(X_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    X_test = X_test.reshape(X_test.shape[0], IMG_ROWS, IMG_COLS, 1)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # train
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_GAN(generator, discriminator)
    train_GAN(generator, discriminator, gan, X_train)
