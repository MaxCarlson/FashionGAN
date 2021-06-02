import os
import numpy as np
import pandas as pd
import keras as K
import tensorflow as tf
import statistics as st
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
desout = layers.Dense(1, activation=K.activations.sigmoid)(x)

descriminator = K.Model(desin, desout)
ganGen.summary()
descriminator.summary()


ganGen.compile()

# Turn off the descriminator weights so it doesn't train when running the GAN
descriminator.trainable = False
GAN = K.models.Sequential(layers=[ganGen, descriminator])
GAN.compile(optimizer=K.optimizers.Adam(learning_rate=0.001, beta_1=0.5), 
        loss=K.losses.binary_crossentropy,
        metrics=['accuracy'])

# Turn them back on se we can train the descriminator
descriminator.trainable = True
descriminator.compile(optimizer=K.optimizers.Adam(learning_rate=0.001, beta_1=0.5), 
        loss=K.losses.binary_crossentropy,
        metrics=['accuracy'])

epochs = 100
batchSize = 1024

dataType = tf.float32
trainData = tf.data.Dataset.from_tensor_slices((tf.cast(trainX, dataType), 
                                                tf.cast(np.ones((trainX.shape[0], 1)), dataType)))
trainData = trainData.shuffle(buffer_size=1024).batch(batchSize//2)

def mvn(size):
    #mvns = np.random.multivariate_normal(mean=np.zeros(100), cov=np.diag(np.ones(100)), 
    #                                     size=(batchSize, 1))
    return np.random.standard_normal((size, generatorInputSize))

ppreds = []
def printPreds(e, ganGen, sz):
    mvns = mvn(sz)
    ppreds.append(ganGen.predict(mvns))
    fig, axs = plt.subplots(1, sz)
    for i in range(sz):
        axs[i].imshow(np.reshape(ppreds[-1][i], (28,28)), cmap='gray')
    plt.savefig(f'e_{e}.jpg')
    plt.close()
    if e != epochs:
        return

    fig, axs = plt.subplots(len(ppreds), sz)
    for i in range(len(ppreds)):
        for j in range(sz):
            axs[i][j].imshow(np.reshape(ppreds[i][j], (28,28)), cmap='gray')
    plt.show()

dls = []
gls = []
for e in range(1, epochs+1):
    edls = []
    egls = []
    print(f'Epoch {e}')
    for b, (X, Y) in enumerate(trainData):
        # Generate images from the generator
        tBatchSize = X.shape[0]
        aBatchSize = tBatchSize * 2
        genImages = ganGen.predict(tf.random.normal((tBatchSize,generatorInputSize)))
        
        tX = tf.concat([X, genImages], axis=0)
        tY = tf.concat([Y, tf.zeros((tBatchSize, 1))], axis=0)

        ld, da = descriminator.train_on_batch(tX, tY)
        edls.append(ld)
        mvns = tf.random.normal((aBatchSize, generatorInputSize))
        lg, ga  = GAN.train_on_batch(x=mvns, y=tf.ones((len(mvns),1)))
        egls.append(lg)

        if b % 5 == 0:
            print(f'genLoss: {lg:.12f}, descLoss: {ld:.12f}, GAN acc:{ga:.4f}, DescFake acc:{da:.4f}')

    if e and e % 10 == 0:
        printPreds(e, ganGen, 5)
    dls.append(st.mean(edls))
    gls.append(st.mean(egls))


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(range(epochs), dls)
ax2.plot(range(epochs), gls)
ax1.set_title('Descriminator Loss')
ax2.set_title('GAN Loss')
plt.show()

