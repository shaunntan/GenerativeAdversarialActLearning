from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
# from extra_keras_datasets import usps
import usps.usps as usps

class RandomTrainer:
    '''
    Perform training by randomly adding new labelled samples from unlabelled pool

    Parameters
    ---
    traindatasettype: string
        'mnist' | 'cifar10'

    testdatasettype: string
        'mnist' | 'usps' | 'cifar10'

    oraclepath: string
        path to trained tensorflow oracle

    n_samples_end: int
        perform Active SVM training until n_samples_end number of samples in labelled set is reached.

    start_samples: int
        no. of starting labeled samples for SVC learner

    Returns
    ---
    ActiveSVMTrainer Class

    Attributes
    ---
    traindatasettype: dataset used for training
    learner_acc_history: SVC accuracy history
    n_samples: number of samples in labelled set at each classifier update
    n_samples_end: target ending number of samples in labelled set
    oracle: instance of Keras.Model of oracle
    oraclepath: path to oracle
    start_samples: starting number of samples in labelled set
    x_test: x testing set
    x_train_labelled: starting labelled x set
    x_train_labelled_final: ending labelled x set
    x_train_unlabelled: starting unlabelled x set
    x_train_unlabelled_final: ending unlabelled x set
    y_test: y of x_test
    y_train_labelled: starting labels of x set
    y_train_labelled_final: ending labels of labelled x set
    y_train_unlabelled: start labels of unlabelled x set
    y_train_unlabelled_final: ending labels of unlabelled x set
    '''
    def __init__(self, traindatasettype, testdatasettype, oraclepath, n_samples_end, start_samples):
        self.traindatasettype = traindatasettype
        self.testdatasettype = testdatasettype
        print('Random trainer')
        if self.traindatasettype == 'mnist' and self.testdatasettype == 'mnist':
            print(f'Training on mnist, testing on mnist')
        elif self.traindatasettype == 'mnist' and self.testdatasettype == 'usps':
            print(f'Training on mnist, testing on usps')
        elif self.traindatasettype == 'cifar10' and self.testdatasettype == 'cifar10':
            print(f'Training and testing with cifar10')
        else:
            raise Exception('Train and test combinations allowed are: 1) mnist & mnist, 2) mnist & usps, 3) cifar10 & cifar10')

        self.n_samples_end = n_samples_end
        self.start_samples = start_samples
        self.oraclepath = oraclepath
        self.oracle = self.loadoracle(oraclepath)
        self.x_train_labelled, self.y_train_labelled, self.x_train_unlabelled, self.y_train_unlabelled, self.x_test, self.y_test = self.loaddata()

        self.x_test = np.squeeze(self.x_test)
        self.x_test = self.x_test.reshape((self.x_test.shape[0],-1))
        self.y_test = self.y_test.ravel()

        self.x_train_labelled_final, self.y_train_labelled_final, self.x_train_unlabelled_final, self.y_train_unlabelled_final, self.learner_acc_history, self.n_samples = self.randomtrainer(self.x_train_labelled, self.y_train_labelled, self.x_train_unlabelled, self.y_train_unlabelled)

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

        # np.random.seed(0)
        rand_idx = np.random.choice(np.arange(x_train.shape[0]), size = self.start_samples, replace = False)
        inverse_idx = [i for i in np.arange(x_train.shape[0]) if i not in rand_idx]
        x_train_labelled = x_train[rand_idx]
        y_train_labelled = y_train[rand_idx]
        x_train_unlabelled = x_train[inverse_idx]
        y_train_unlabelled = y_train[inverse_idx]

        return x_train_labelled, y_train_labelled, x_train_unlabelled, y_train_unlabelled, x_test, y_test

    def trainsvc(self, xtrain, ytrain):
        x = np.squeeze(xtrain)
        xt = x.reshape((x.shape[0],-1))
        svc = SVC(kernel='linear', probability = True, gamma=0.001).fit(xt, ytrain.ravel())
        return svc

    def getidxofnearesttoplane(self, svc, unlabelleddata):
        unlabelleddata = np.squeeze(unlabelleddata)
        unlabelleddata = unlabelleddata.reshape((unlabelleddata.shape[0],-1))
        
        w = svc.coef_
        b = svc.intercept_
        w_norm = np.linalg.norm(w)
        dist_list = []

        for r in range(unlabelleddata.shape[0]):
            d = np.abs(np.dot(w,unlabelleddata[r]) + b)/w_norm
            dist_list.append(d)
        
        idx_nearest_to_hyperplane = np.argmin(dist_list)
        
        return idx_nearest_to_hyperplane

    def loadoracle(self, oraclepath):
        oracle = load_model(oraclepath)
        oracle.trainable = False
        return oracle

    def randomtrainer(self, x_train_labelled, y_train_labelled, x_train_unlabelled, y_train_unlabelled):
        accuracy = []
        n_samples = []

        n_samples_this = x_train_labelled.shape[0]
        n_samples.append(n_samples_this)

        # train first svc
        svc = self.trainsvc(x_train_labelled, y_train_labelled)
        
        # get test set accuracy
        test_preds = svc.predict(self.x_test)
        acc = accuracy_score(self.y_test, test_preds)
        accuracy.append(acc)
        
        print(f'Starting no. of samples in labelled set: {n_samples_this}')
        print(f'Starting no. of samples in unlabelled set: {x_train_unlabelled.shape[0]}')
        print(f'Starting SVC Acc: {acc}')

        while n_samples_this < self.n_samples_end:

            cnt = 0
            while cnt < 10:
                # get randomly pick a sample from unlabelled pool
                random_idx = np.random.randint(0, high = x_train_unlabelled.shape[0])

                # get random sample
                rnd_label = np.reshape(y_train_unlabelled[random_idx], (1,1))
                rnd_labelled_img = np.expand_dims(x_train_unlabelled[random_idx], axis = 0)
                # add to labelled set
                y_train_labelled = np.concatenate((y_train_labelled, rnd_label), axis = 0)
                x_train_labelled = np.concatenate((x_train_labelled, rnd_labelled_img), axis = 0)
                # delete from unlabelled set
                x_train_unlabelled = np.delete(x_train_unlabelled, random_idx, axis = 0)
                y_train_unlabelled = np.delete(y_train_unlabelled, random_idx, axis = 0)
                cnt += 1
            
            n_samples_this = x_train_labelled.shape[0]
            n_samples.append(n_samples_this)

            # update SVC
            svc = self.trainsvc(x_train_labelled, y_train_labelled)
            
            # get test set accuracy
            test_preds = svc.predict(self.x_test)
            acc = accuracy_score(self.y_test, test_preds)
            accuracy.append(acc)
            
            print(f'No. of samples in training set: {n_samples_this}')
            print(f'SVC Acc: {acc}')
        
        print(f'Ending no. of samples in unlabelled set: {x_train_unlabelled.shape[0]}')

        return x_train_labelled, y_train_labelled, x_train_unlabelled, y_train_unlabelled, accuracy, n_samples