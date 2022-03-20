import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class OracleTrainer:
    '''
    Train Oracles

    Parameters
    ---
    datasettype: string
        'mnist' | 'cifar10'

    pretrainedtype: string
        'vgg16'

    model_name: string
        e.g.: 'ciar10oracle'
        the name of the model to be trained. Also used as filename.
    
    savepath: string
        e.g.: './oracles/'
        the folder path where the trained oracle will be saved. in combination with model_name, the trained oracle will be saved in: f'{savepath} + {model_name}.h5'

    lr: float
        learning rate when training oracle

    epochs: int
        max number of epochs to train oracle for. training will stop when, earlystopping_patience or epochs is reached, whichever is earlier.

    batchsize: int
        batch size during training of oracle

    earlystopping_patience: int
        number of epochs where val_accuracy does not increase and training will be stopped

    Return
    ---
    OracleTrainer Class

    Attributes
    ---
    basemodel: Keras.Model instance of the pretrained model that was used
    batchsize: the batch size used when training the oracle
    datasettype: dataset type of the trained oracle
    earlystopping_patience: training is stopped when this number of epochs did not have val_accuracy increase
    epochs: max number of epochs the oracle was trained for
    imagedatagenerator: the image generator used when training the oracle
    lr: learning rate when training the oracle
    model_name: oracle model name
    oraclemode: Kears.Model instance of teh trained oracle
    pretrainedtype: pretrained model type
    savepath: folder path where the trained oracle is saved
    x_test: the x of testing dataset
    x_train: the x of training dataset
    y_testL the bales for testing dataset
    y_train: the labels for training dataset
    '''
    def __init__(self, datasettype, pretrainedtype, model_name, savepath, lr, epochs, batchsize, earlystopping_patience):

        self.datasettype = datasettype
        self.pretrainedtype = pretrainedtype
        self.model_name = model_name
        self.savepath = savepath
        self.lr = lr
        self.epochs = epochs
        self.batchsize = batchsize
        self.earlystopping_patience = earlystopping_patience

        self.x_train, self.y_train, self.x_test, self.y_test = self.loaddata()
        self.basemodel = self.loadpretrained()
        self.oraclemodel = self.createmodel()
        self.imagedatagenerator = self.imagedatagen()

        self.fitmodel()
        self.getaccuracy()

    def loaddata(self):
        if self.datasettype == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            # 5 = Class 0, 7 = Class 1
            x_train = x_train[(y_train == 5) | (y_train == 7),]
            y_train = y_train[(y_train == 5) | (y_train == 7)]
            y_train = np.where(y_train == 5, 0, 1).reshape((-1,1))

            x_test = x_test[(y_test == 5) | (y_test == 7),]
            y_test = y_test[(y_test == 5) | (y_test == 7)]
            y_test = np.where(y_test == 5, 0, 1).reshape((-1,1))

            x_train = np.expand_dims(x_train, axis = 3)
            x_train = np.repeat(x_train,3,axis = 3)
            x_train = tf.image.resize(x_train, [32,32])/255

            x_test = np.expand_dims(x_test, axis = 3)
            x_test = np.repeat(x_test,3,axis = 3)
            x_test = tf.image.resize(x_test, [32,32])/255

        elif self.datasettype == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            y_train = np.squeeze(y_train)
            y_test = np.squeeze(y_test)
            # Automobile(1) = Class 0, Horse(7) = Class 1
            x_train = x_train[(y_train == 1) | (y_train == 7),]
            y_train = y_train[(y_train == 1) | (y_train == 7)]
            y_train = np.where(y_train == 1, 0, 1).reshape((-1,1))

            x_test = x_test[(y_test == 1) | (y_test == 7),]
            y_test = y_test[(y_test == 1) | (y_test == 7)]
            y_test = np.where(y_test == 1, 0, 1).reshape((-1,1))

            x_train = x_train/255
            x_test = x_test/255

        return x_train, y_train, x_test, y_test

    def loadpretrained(self):
        if self.pretrainedtype == 'vgg16':
            base_model = VGG16(input_shape = (self.x_train[0].shape[0],self.x_train[0].shape[1],self.x_train[0].shape[2]),
                    include_top = False, weights = 'imagenet')

        return base_model

    def createmodel(self):
        model_name = self.model_name

        inputs = tf.keras.Input(shape=(self.x_train[0].shape[0],self.x_train[0].shape[1],self.x_train[0].shape[2]))
        x = self.basemodel(inputs, training=False)
        x = Flatten()(x)
        outputs = Dense(1, activation = 'sigmoid')(x)
        model = tf.keras.Model(inputs, outputs, name = model_name)
        model.summary()

        return model

    def imagedatagen(self):
        imagedatagenerator = ImageDataGenerator(
            rotation_range=20,
            shear_range=10, validation_split = 0.2)

        imagedatagenerator.fit(self.x_train)
        return imagedatagenerator

    def fitmodel(self):
        self.filepath = f"{self.savepath + self.model_name}.h5"

        self.oraclemodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    metrics=['accuracy'])

        earlystop = EarlyStopping(monitor='val_accuracy', patience = self.earlystopping_patience , restore_best_weights=True)
        # checkpoint = ModelCheckpoint(path, monitor = 'val_accuracy', save_best_only=True)

        callbacks = [earlystop]

        self.oraclemodel.fit(self.imagedatagenerator.flow(self.x_train, self.y_train, batch_size = self.batchsize, subset = 'training'),
                                validation_data=self.imagedatagenerator.flow(self.x_train, self.y_train, batch_size = self.batchsize, subset = 'validation'),
                                steps_per_epoch=len(self.x_train)*0.8/self.batchsize,
                                epochs = self.epochs, 
                                callbacks = callbacks)

        self.oraclemodel.save(self.filepath, save_format = 'h5')

    def getaccuracy(self):
        model = load_model(self.filepath)
        preds = model.predict(self.x_test)
        preds = np.squeeze(np.round(preds, 0).astype('int'))
        self.acc = np.mean(preds == np.squeeze(self.y_test))
        print(f'Oracle Accuracy on Test Set: {self.acc}')