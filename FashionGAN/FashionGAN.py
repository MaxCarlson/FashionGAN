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

trainX = np.reshape(trainX, (trainX.shape[0], 28, 28))
testX = np.reshape(testX, (testX.shape[0], 28, 28))


generatorInputSize = 100
genin = K.Input(shape=(generatorInputSize))
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
genout = layers.Reshape((28,28))(x)

ganGen = K.Model(genin, genout)

desin = K.Input(shape=(28,28))
x = layers.Reshape((784,))(desin)
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
desout = layers.BatchNormalization()(x)

descriminator = K.Model(desin, desout)
ganGen.summary()
descriminator.summary()


descriminator.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        loss=K.losses.binary_crossentropy,
        metrics=[])
ganGen.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        loss=K.losses.binary_crossentropy,
        metrics=[])

GAN = K.models.Sequential(layers=[ganGen, descriminator])
GAN.compile(optimizer=K.optimizers.Adam(learning_rate=0.001), 
        loss=K.losses.binary_crossentropy,
        metrics=[])


epochs = 100
batchSize = 1024

dataType = tf.float32

trainData = tf.data.Dataset.from_tensor_slices((tf.cast(trainX, dataType), 
                                                tf.cast(np.ones((trainX.shape[0], 1)), dataType)))
trainData = trainData.shuffle(buffer_size=1024).batch(batchSize)

def mvn(size):
    #mvns = np.random.multivariate_normal(mean=np.zeros(100), cov=np.diag(np.ones(100)), 
    #                                     size=(batchSize, 1))
    return np.random.standard_normal((size, generatorInputSize))

def printPreds(ganGen, sz):
    mvns = mvn(sz)
    fig, axs = plt.subplots(1, sz)
    preds = ganGen.predict(mvns)
    for i in range(sz):
        axs[i].imshow(np.reshape(preds[i], (28,28)), cmap='gray')
    plt.show()

for e in range(epochs):

    for b, (X, Y) in enumerate(trainData):
        
        tBatchSize = X.shape[0]
        #mvns = mvn(tBatchSize)

        # Generate images from the generator
        genImages = ganGen.predict(tf.random.normal((tBatchSize,generatorInputSize)))
        
        tX = tf.concat([X, genImages], axis=0)
        tY = tf.concat([Y,tf.zeros((tBatchSize, 1))], axis=0)

        #genImages = ganGen.predict(mvns)     
        # Concat generated images with real  
        # then while preserving shuffle while preserving ordering
        #tX = np.concatenate([np.reshape(X.numpy(), (tBatchSize,28,28)), genImages])
        #tY = np.concatenate([np.reshape(Y.numpy(), (tBatchSize)), np.zeros((tBatchSize))])
        #tX, tY = shuffle(tX, tY, random_state=0)

        # Train the descriminator
        descriminator.trianable = True
        #trf = tf.data.Dataset.from_tensor_slices((tX, tY))
        #trf = trainData.shuffle(buffer_size=1024).batch(tBatchSize)
        print('Training Descriminator')
        descriminator.fit(tX, tY, batch_size=tBatchSize, shuffle=True)

        # Train the generator
        descriminator.trianable = False
        
        mvns = tf.random.normal((tBatchSize,generatorInputSize))
        print('Training Generator')
        GAN.fit(x=mvns, y=tf.ones(len(mvns)), batch_size=tBatchSize, shuffle=True)


    if e and e % 3 == 0:
        printPreds(ganGen, 5)
