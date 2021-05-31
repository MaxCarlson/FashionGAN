import os
import numpy as np
import pandas as pd
import keras as K
import tensorflow as tf
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

generatorInputSize = 100
genin = K.Input(shape=(generatorInputSize))
genlayers = []
x = layers.Dense(256)(genin)
genlayers.append(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512)(x)
genlayers.append(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(648)(x)
genlayers.append(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(784)(x)
genlayers.append(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
genout = layers.Reshape((28,28,1))(x)

ganGen = K.Model(genin, genout)

deslayers = []
desin = K.Input(shape=(28,28,1))
x = layers.Reshape((784,))(desin)
x = layers.Dense(784)(x)
deslayers.append(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512)(x)
deslayers.append(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256)(x)
deslayers.append(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(1)(x)
deslayers.append(x)
x = K.activations.sigmoid(x)
desout = layers.BatchNormalization()(x)

ganDescrim = K.Model(desin, desout)
ganGen.summary()
ganDescrim.summary()

GAN = K.models.Sequential(layers=[ganGen, ganDescrim])

GAN.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        loss=K.losses.binary_crossentropy,
        metrics=[])


epochs = 2
batchSize = 32
optim = K.optimizers.Adam()
lossfn = K.losses.BinaryCrossentropy()

#GAN.fit(d1, d2, batchSize, epochs)

trainData = tf.data.Dataset.from_tensor_slices((trainX, np.ones((trainX.shape[0], 1))))

for e in range(epochs):

    for b, (X, Y) in enumerate(trainData):
        #mvns = np.random.multivariate_normal(mean=np.zeros(100), cov=np.diag(np.ones(100)), 
        #                                     size=(batchSize, 1))
        mvns = np.random.rand(batchSize, generatorInputSize)

        # Generate images from the generator
        genImages = ganGen.predict(mvns)        





        a=5
