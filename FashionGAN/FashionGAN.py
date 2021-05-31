import os
import numpy as np
import pandas as pd
import keras as K
import tensorflow as tf
from sklearn.utils import shuffle
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
genout = layers.Reshape((28,28))(x)

ganGen = K.Model(genin, genout)

deslayers = []
desin = K.Input(shape=(28,28))
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

descriminator = K.Model(desin, desout)
ganGen.summary()
descriminator.summary()

GAN = K.models.Sequential(layers=[ganGen, descriminator])

GAN.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        loss=K.losses.binary_crossentropy,
        metrics=[])

descriminator.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        loss=K.losses.binary_crossentropy,
        metrics=[])
ganGen.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        loss=K.losses.binary_crossentropy,
        metrics=[])

epochs = 2
batchSize = 512

trainData = tf.data.Dataset.from_tensor_slices((trainX, np.ones((trainX.shape[0], 1))))
trainData = trainData.shuffle(buffer_size=1024).batch(batchSize)

for e in range(epochs):

    for b, (X, Y) in enumerate(trainData):
        #mvns = np.random.multivariate_normal(mean=np.zeros(100), cov=np.diag(np.ones(100)), 
        #                                     size=(batchSize, 1))
        mvns = np.random.rand(batchSize, generatorInputSize)

        # Generate images from the generator
        genImages = ganGen.predict(mvns)        

        # Concat generated images with real  
        # then while preserving shuffle while preserving ordering
        tX = np.concatenate([np.reshape(X.numpy(), (batchSize,28,28)), genImages])
        tY = np.concatenate([np.reshape(Y.numpy(), (batchSize)), np.zeros((batchSize))])
        #tX, tY = shuffle(tX, tY, random_state=0)

        # Train the descriminator
        descriminator.trianable = True

        trf = tf.data.Dataset.from_tensor_slices((tX, tY))
        trf = trainData.shuffle(buffer_size=1024).batch(batchSize)

        descriminator.fit(tX, tY, batch_size=batchSize)

        # Train the generator
        descriminator.trianable = False

        mvns = np.random.rand(batchSize*2, generatorInputSize)
        GAN.fit(x=mvns, y=np.ones(batchSize*2), batch_size=batchSize)



