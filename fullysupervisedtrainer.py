from sklearn.svm import SVC
import tensorflow as tf
import numpy as np
# from extra_keras_datasets import usps
import usps.usps as usps

class FSTrainer:
    '''
    Perform fully supervised training on train set and return test set accuracy

    Parameters
    ---
    traindatasettype: string
        'mnist' | 'cifar10'

    testdatasettype: string
        'mnist' | 'usps' | 'cifar10'

    Returns
    ---
    FSTrainer Class

    Attributes
    ---
    traindatasettype: dataset used for training
    testdatasettype: dataset used for testing
    learner_acc_history: SVC accuracy history
    x_test: x of testing set
    x_train: x of train set
    y_test: y of x_test
    y_train: y of x_train
    '''
    def __init__(self, traindatasettype, testdatasettype):
        self.traindatasettype = traindatasettype
        self.testdatasettype = testdatasettype
        print('Fully Supervised trainer')
        if self.traindatasettype == 'mnist' and self.testdatasettype == 'mnist':
            print(f'Training on mnist, testing on mnist')
        elif self.traindatasettype == 'mnist' and self.testdatasettype == 'usps':
            print(f'Training on mnist, testing on usps')
        elif self.traindatasettype == 'cifar10' and self.testdatasettype == 'cifar10':
            print(f'Training and testing with cifar10')
        else:
            raise Exception('Train and test combinations allowed are: 1) mnist & mnist, 2) mnist & usps, 3) cifar10 & cifar10')

        self.x_train, self.y_train, self.x_test, self.y_test = self.loaddata()

        self.svc, self.learner_acc_history = self.trainsvc(self.x_train, self.y_train, self.x_test, self.y_test)

    def loaddata(self):

        if self.traindatasettype == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = (x_train[(y_train == 5) | (y_train == 7),]- 127.5) / 127.5
            # x_train = np.expand_dims(x_train, axis=3)
            y_train = y_train[(y_train == 5) | (y_train == 7)]
            y_train = np.where(y_train == 5, 0, 1).reshape((-1,1))

            if self.testdatasettype == 'mnist':
                x_test = (x_test[(y_test == 5) | (y_test == 7),]- 127.5) / 127.5
                # x_test = np.expand_dims(x_test, axis=3)
                y_test = y_test[(y_test == 5) | (y_test == 7)]
                y_test = np.where(y_test == 5, 0, 1).reshape((-1,1))

            elif self.testdatasettype == 'usps':
                (x_test, y_test), (_, _) = usps.load_data()

                y_test = y_test.astype('int')
                x_test = x_test[(y_test == 5) | (y_test == 7),]
                x_test = np.expand_dims(x_test, axis=3)
                x_test = (x_test + 1) / 2
                x_test = tf.image.pad_to_bounding_box(x_test,4,4,24,24)
                x_test = tf.image.resize(x_test, [28,28])
                x_test = (x_test * 2) - 1
                x_test = np.squeeze(x_test)

                y_test = y_test[(y_test == 5) | (y_test == 7)]
                y_test = np.where(y_test == 5, 0, 1).reshape((-1,1))

        elif self.traindatasettype == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            y_train = np.squeeze(y_train)
            y_test = np.squeeze(y_test)
            x_train = (x_train[(y_train == 1) | (y_train == 7),]- 127.5) / 127.5
            y_train = y_train[(y_train == 1) | (y_train == 7)]
            y_train = np.where(y_train == 1, 0, 1).reshape((-1,1))

            x_test = (x_test[(y_test == 1) | (y_test == 7),]- 127.5) / 127.5
            y_test = y_test[(y_test == 1) | (y_test == 7)]
            y_test = np.where(y_test == 1, 0, 1).reshape((-1,1))

        return x_train, y_train, x_test, y_test

    def trainsvc(self, xtrain, ytrain, xtest, ytest):
        x_train = np.squeeze(xtrain)
        x_train = x_train.reshape((x_train.shape[0],-1))

        x_test = np.squeeze(xtest)
        x_test = x_test.reshape((x_test.shape[0],-1))
        print(f'xtrain shape: {x_train.shape}')
        print(f'xtest shape: {x_test.shape}')
        print(f'ytrain shape: {ytrain.shape}')
        print(f'ytrain shape: {ytest.shape}')
        svc = SVC(kernel='linear', gamma=0.001).fit(x_train, ytrain.ravel())

        y_test = np.squeeze(ytest)
        test_acc = svc.score(x_test, y_test)

        return svc, test_acc