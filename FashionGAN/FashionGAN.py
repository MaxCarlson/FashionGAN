import os
import numpy as np
import pandas as pd
import keras as K
import matplotlib.pyplot as plt
from keras import models, layers
from sklearn.decomposition import PCA
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28, 1)

    return np.copy(images), np.copy(labels)

trainX, trainY = load_mnist('data', kind='train')
testX, testY = load_mnist('data', kind='t10k')

trainX = np.divide(trainX, 255)
testX = np.divide(testX, 255)


genin = K.Input(shape=(100))
x = layers.Dense(256)(genin)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512)(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(648)(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(784)(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
genout = layers.Reshape((28,28,1))(x)

ganGen = K.Model(genin, genout)

descrimin = K.Input(shape=(28,28,1))
x = layers.Reshape((784,))(descrimin)
x = layers.Dense(784)(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512)(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(1)(x)
x = K.activations.sigmoid(x)
descrimout = layers.BatchNormalization()(x)

ganDescrim = K.Model(descrimin, descrimout)

ganGen.summary()
ganDescrim.summary()

GAN = K.layers.Concatenate()([ganGen, ganDescrim])

