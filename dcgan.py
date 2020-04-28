'''
Useful doc:
- https://arxiv.org/pdf/1511.06434.pdf
- https://keras.io/layers/
- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''

# Native libs
import os
import time
import logging

# Added libs
import tensorflow as tf # Machine learning lib
import numpy as np
import matplotlib.pyplot as plt

# --------------
# Global variables
# --------------

global LOGGER

# --------------
# Data processing
# --------------

def load_data():
    """
    Returns (x_train, y_train), (x_test, y_test)

    x: Each pixel of each image in the dataset is a number between 0 and
    255.
    y: Label of each image
    """

    LOGGER.info('Loading data')
    return tf.keras.datasets.mnist.load_data()

def cast_and_normalize(images):
    # Cast the images to float32 (pixels)
    # images.shape[0] is the number of images
    images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
    # Normalize the images between -1 and 1 from [0, 255] (pixels)
    images = (images - 127.5) / 127.5
    return images

def to_dataset(images, buffer_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images)
    # Creates a chunk
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset

# --------------
# File & image management
# --------------

def generate_and_save_images(model, epoch, test_input):
    # Training is set to False so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


# --------------
# Models
# --------------

def generator_model():
    """
    This model is based on this paper: https://arxiv.org/pdf/1511.06434.pdf
    """

    # This is a bit complicated. I'll try to explain it as munch as I can. Make
    # sure to take a look at the figure 1 in the paper to understand what I'm
    # going to say

    # The generator model is a neural network (NN). We will use keras to build
    # a NN.

    # It creates a linear stack of layers. Then we'll be able to add "layers".
    model = tf.keras.Sequential()

    # Let's build the first layer. The input layer.
    # The dense layer is the regular NN layer. output = activation(input dot weigth + bias)
    # 256*7*7: In the paper, they wanted to create a 64x64 pixel image. So they
    # started with 1024x4x4, so it goes to 512x8x8 -> 256x16x16 -> 128x32x32 ->
    # 64x64x3. While always using stride=2. We want the generator to create a
    # 28x28 image. So we will start at 256x7x7 so that it goes to 128x7x7 (with
    # a stride of 1) -> 64x14x14 -> 64x28x28. The stride is the steps of the
    # convolution along the height and the width.
    # bias=False: We don't want the generator to use bias in its calculations
    # input_shape: According to the paper, we need to give a 100x1 noise vector
    model.add(tf.keras.layers.Dense(256*7*7, use_bias=False, input_shape=(100,)))

    # This layer uses batch normalization and ReLu activation
    # It is needed to help with the flow of gradients during training. It is a
    # critical contribution of the DCGAN paper
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # We need to reshape because the shape of the data will be (-, 4, 4, 1024)
    # That's because we gave a specific input_shape of (100,)
    model.add(tf.keras.layers.Reshape((7, 7, 256)))

    assert model.output_shape == (None, 7, 7, 256)

    # 128: It is the dimensionality of the output space
    # (5, 5): It specifies the height and the width of the 2D convolution window
    # (1, 1): It specifies the steps (x, y) of the convolution along the height
    #           and width.
    # 'same': We keep the same padding. The padding can be specified by
    #           output_padding, but we don't need it.
    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Same explanations as before
    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Output layer
    # Basically the same thing, but with a different activation. The output of
    # the generator is fed through a tanh function to return it to the input
    # data range of [−1, 1]
    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def discriminator_model():
    # See the comments in generator_model to understand the concepts
    model = tf.keras.Sequential()

    # 64x28x28
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    # Dropout consists in randomly setting a fraction rate of input units to 0
    # at each update during training time, which helps prevent overfitting.
    model.add(tf.keras.layers.Dropout(0.3))

    # 128x14x14
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    # 256x7x7 TODO: needed?
    # model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    # model.add(tf.keras.layers.LeakyReLU())
    # model.add(tf.keras.layers.Dropout(0.3))

    # Flattens the input. Does not affect the batch size. If inputs are shaped
    # (batch,) without a channel dimension, then flattening adds an extra
    # channel dimension and output shapes are (batch, 1).
    # 256x1x1
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

# --------------
# Loss functions
# --------------

def discriminator_loss(loss_func, real_output, fake_output):
    real_loss = loss_func(tf.ones_like(real_output), real_output)
    fake_loss = loss_func(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(loss_func, fake_output):
    """
    The generator’s loss quantifies how well it was able to trick the
    discriminator. Intuitively, if the generator is performing well, the
    discriminator will classify the fake images as real (or 1).
    """
    return loss_func(tf.ones_like(fake_output), fake_output)

# --------------
# Training
# --------------

@tf.function
def train(images, buffer_size, noise_dim, gen, dis, gen_opt, dis_opt, loss_func):
    noise = tf.random.normal([buffer_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise, training=True)

        real_output = dis(images, training=True)
        fake_output = dis(generated_images, training=True)

        gen_loss = generator_loss(loss_func, fake_output)
        disc_loss = discriminator_loss(loss_func, real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, dis.trainable_variables)

        gen_opt.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
        dis_opt.apply_gradients(zip(gradients_of_discriminator, dis.trainable_variables))

# --------------
# Main
# --------------

def main():
    # Parameters
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    EPOCHS = 10
    NOISE_DIM = 100 # Based on DCGAN paper
    SAMPLES = 5
    # We define a specific seed here to show the evolution of the generator
    WATCHED_SEED = tf.random.normal([SAMPLES, NOISE_DIM])

    # Load the data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # Turn the data into a tensorflow dataset
    train_dataset = to_dataset(cast_and_normalize(train_images), BUFFER_SIZE, BATCH_SIZE)

    # Create models
    generator = generator_model()
    discriminator = discriminator_model()

    # Loss function (cross entropy loss)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Optimizer
    # Why do we need an optimizer?? The discriminator and the generator
    # optimizers are different since we will train two networks, separately.
    # The Adam optimization algorithm is an extension of stochastic gradient
    # descent. Stochastic gradient descent maintains a single learning rate
    # (termed alpha) for all weight updates while the learning rate does not
    # change during training. A learning rate is maintained for each network
    # weight (parameter) and adapts as learning unfolds
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Train
    for epoch in range(EPOCHS):
        start = time.time()

        for image in train_dataset:
            train(
                image,
                BUFFER_SIZE, NOISE_DIM,
                generator, discriminator,
                generator_optimizer, discriminator_optimizer,
                cross_entropy
            )

            generate_and_save_images(generator, epoch + 1, WATCHED_SEED)

            LOGGER.info('Epoch %i, %i sec', epoch+1, time.time()-start)

    # Generate after the final epoch
    generate_and_save_images(generator, EPOCHS, WATCHED_SEED)



if __name__ == '__main__':
    # Setting up the logger
    global LOGGER
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s ' +
                                  '-- %(levelname)s ' +
                                  '-- [%(filename)s:%(lineno)s ' +
                                  '-- %(funcName)s() ] ' +
                                  '-- %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    main()