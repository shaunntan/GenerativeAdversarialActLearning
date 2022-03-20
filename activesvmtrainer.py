from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

class ActiveSVMTrainer:
    '''
    Perform Active SVM Training

    Parameters
    ---
    datasettype: string
        'mnist' | 'cifar10'

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
    datasettype: dataset used for training
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
    def __init__(self, datasettype, oraclepath, n_samples_end, start_samples):
        self.datasettype = datasettype
        self.n_samples_end = n_samples_end
        self.start_samples = start_samples
        self.oraclepath = oraclepath
        self.oracle = self.loadoracle(oraclepath)
        self.x_train_labelled, self.y_train_labelled, self.x_train_unlabelled, self.y_train_unlabelled, self.x_test, self.y_test = self.loaddata()

        self.x_test = np.squeeze(self.x_test)
        self.x_test = self.x_test.reshape((self.x_test.shape[0],-1))
        self.y_test = self.y_test.ravel()

        self.x_train_labelled_final, self.y_train_labelled_final, self.x_train_unlabelled_final, self.y_train_unlabelled_final, self.learner_acc_history, self.n_samples = self.activesvmtrainer(self.x_train_labelled, self.y_train_labelled, self.x_train_unlabelled, self.y_train_unlabelled)

    def loaddata(self):

        if self.datasettype == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = (x_train[(y_train == 5) | (y_train == 7),]- 127.5) / 127.5
            # x_train = np.expand_dims(x_train, axis=3)
            y_train = y_train[(y_train == 5) | (y_train == 7)]
            y_train = np.where(y_train == 5, 0, 1).reshape((-1,1))

            x_test = (x_test[(y_test == 5) | (y_test == 7),]- 127.5) / 127.5
            # x_test = np.expand_dims(x_test, axis=3)
            y_test = y_test[(y_test == 5) | (y_test == 7)]
            y_test = np.where(y_test == 5, 0, 1).reshape((-1,1))

        elif self.datasettype == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            y_train = np.squeeze(y_train)
            y_test = np.squeeze(y_test)
            x_train = (x_train[(y_train == 1) | (y_train == 7),]- 127.5) / 127.5
            y_train = y_train[(y_train == 1) | (y_train == 7)]
            y_train = np.where(y_train == 1, 0, 1).reshape((-1,1))

            x_test = (x_test[(y_test == 1) | (y_test == 7),]- 127.5) / 127.5
            y_test = y_test[(y_test == 1) | (y_test == 7)]
            y_test = np.where(y_test == 1, 0, 1).reshape((-1,1))

        np.random.seed(0)
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

    def activesvmtrainer(self, x_train_labelled, y_train_labelled, x_train_unlabelled, y_train_unlabelled):
        accuracy = []
        n_samples = []

        n_samples_this = x_train_labelled.shape[0]
        i = 0
        while n_samples_this < self.n_samples_end:
            print(f'Round {i+1}')
            # update SVC
            n_samples.append(x_train_labelled.shape[0])
            svc = self.trainsvc(x_train_labelled, y_train_labelled)
            
            # get test set accuracy
            test_preds = svc.predict(self.x_test)
            acc = accuracy_score(self.y_test, test_preds)
            accuracy.append(acc)
            print(f'SVC Acc: {acc}')
            cnt = 0
            while cnt < 10:
                # get index of most uncertain sample
                unsure_idx = self.getidxofnearesttoplane(svc, x_train_unlabelled)
                unsure_img = x_train_unlabelled[unsure_idx]
                # use oracle to predict uncertain sample
                if self.datasettype == 'mnist':
                    unsure_img = np.expand_dims(unsure_img, axis = 0)
                    unsure_img = np.expand_dims(unsure_img, axis = 3)
                    unsure_img = np.repeat(unsure_img, 3, axis = 3)
                    unsure_img = (tf.image.resize(unsure_img, [32,32]) + 1)/2

                elif self.datasettype == 'cifar10':
                    unsure_img = np.expand_dims(unsure_img, axis = 0)

                oracle_label = np.squeeze(self.oracle.predict(unsure_img))

                # get label of uncertain sample
                new_label = np.reshape(np.round(oracle_label).astype('int'), (1,1))
                new_labelled_img = np.expand_dims(x_train_unlabelled[unsure_idx], axis = 0)
                # add to labelled set
                y_train_labelled = np.concatenate((y_train_labelled, new_label), axis = 0)
                x_train_labelled = np.concatenate((x_train_labelled, new_labelled_img), axis = 0)
                # delete from unlabelled set
                x_train_unlabelled = np.delete(x_train_unlabelled, unsure_idx, axis = 0)
                y_train_unlabelled = np.delete(y_train_unlabelled, unsure_idx, axis = 0)
                cnt += 1
            i += 1

            n_samples_this = x_train_labelled.shape[0]
            print(f'No. of samples in training set: {n_samples_this}')

        return x_train_labelled, y_train_labelled, x_train_unlabelled, y_train_unlabelled, accuracy, n_samples