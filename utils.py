import random
import scipy
import numpy as np
from keras import models
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dense, Activation, Flatten, Reshape, Dropout
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as k

KERNEL_INITIALIZER = 'glorot_uniform'
BN_MOMENTUM = 0.3
ALPHA_D = 0.2
ALPHA_G = 0.2

def build_generator(noise_shape):
    """Create Generator
    Returns:
      Keras Model
    """
    def deconv2d(x, filters, shape=(4, 4)):
        """Helpful function to prevent copying code
        """ 
        x = Conv2DTranspose(filters, shape, padding='same',
                            strides=(2, 2), kernel_initializer=KERNEL_INITIALIZER)(x)
        x = BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = LeakyReLU(alpha=ALPHA_G)(x)
        return x

    noise = Input(noise_shape)
    x = noise
    x = Conv2DTranspose(512, (4, 4),
                        kernel_initializer=KERNEL_INITIALIZER)(x)
    x = BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = LeakyReLU(alpha=ALPHA_G)(x)
    x = deconv2d(x, 256)
    x = deconv2d(x, 128)
    x = deconv2d(x, 64)

    x = Conv2D(64, (3, 3), padding='same',
               kernel_initializer=KERNEL_INITIALIZER)(x)
    x = BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = LeakyReLU(alpha=ALPHA_G)(x)

    x = Conv2DTranspose(4, (4, 4), padding='same', activation='tanh',
                        strides=(2, 2), kernel_initializer=KERNEL_INITIALIZER)(x)

    return models.Model(inputs=noise, outputs=x)



def build_discriminator(shape):
    """Create Discriminator
    Returns:
      Keras Model
    """
    def conv2d(x, filters, shape=(4, 4), **kwargs):
        """Helpful function to prevent copying code
        """ 
        x = Conv2D(filters, shape, strides=(2, 2),
                   padding='same',
                   kernel_initializer=KERNEL_INITIALIZER,
                   **kwargs)(x)
        x = BatchNormalization(momentum=BN_MOMENTUM)(x)
        x = LeakyReLU(alpha=ALPHA_D)(x)
        return x

    inputs = Input(shape=shape)
    x = inputs
    x = Conv2D(64, (4, 4), strides=(2, 2),
               padding='same',
               kernel_initializer=KERNEL_INITIALIZER)(x)
    x = LeakyReLU(alpha=ALPHA_D)(x)
    
    x = conv2d(x, 128)
    x = conv2d(x, 256)
    x = conv2d(x, 512)
    x = Flatten()(x)
    # 1 when "real", 0 when "fake".
    x = Dense(1, activation='sigmoid',
              kernel_initializer=KERNEL_INITIALIZER)(x)
    return models.Model(inputs=inputs, outputs=x)


def make_noise(batch_size,shape):
    """Create matrix with Normal distribution.
    Args:
      batch_size: number of noise vectors.
      shape: shape of noise vector.
    """
    noise = np.random.normal(size=((batch_size,) + shape),
                            loc=0.0,
                            scale=0.1)
    return noise    


def sample_fake(gen,batch_size,noise_shape):
    """Return predictions of generator. Generator input is noise vectors.
    Args:
      gen: generator model.
      batch_size: size of a batch.
      noise_shape: shape of a noise vector.
    Returns:
      predictions: output of gen.
      noise: matrix contains noise vectors.
    """
    noise = make_noise(batch_size,noise_shape)
    predictions = gen.predict(noise)
    return predictions, noise


def sample_faces(faces,batch_size):
    indecies = np.random.choice(len(faces),batch_size)
    faces[indecies]
    reals = np.array(faces[indecies])
    return reals.astype('float32')


def denormalize(images):
    """Does opposite of normalize:
    [-1, 1] to [0, 255].
    """
    images += 1.0 # in [0, 2]
    images *= 128.0 # in [0, 255]
    return images.astype(np.uint8)

def normalize(images):
    """Normalize input images in [-1,1] range.
    Args:
      images: NumPy array
    """
    images = images.astype(np.float32)
    images /= 128.0
    images -= 1.0 # now in [-1, 1]
    return images



#Functions to save images
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def inverse_transform(images):
    return (images + 1.) / 2.
