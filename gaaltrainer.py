import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import load_model
# from extra_keras_datasets import usps
import usps.usps as usps

class GAALTrainer:
    '''
    Perform GAAL training

    Parameters
    ---
    traindatasettype: string
        'mnist' | 'cifar10'

    testdatasettype: string
        'mnist' | 'usps' | 'cifar10'

    generatorpath: string
        path to trained tensorflow generator

    oraclepath: string
        path to trained tensorflow oracle

    n_samples_end: int
        Perform GAAL training until n_samples_end number of samples in labelled set is reached. GAAL training involves the following steps:
            1) Update SVC learner, get accuracy on test set
            2) Generate latents using GD for generating of fake samples
            3) Use oracle to predict labels of fake samples
            4) Subject to threshold, add good samples to training set

    threshold: float
        threshold for good samples, e.g. 1e-8

    start_samples: int
        no. of starting labeled samples for SVC learner

    latent_dim: int
        no. of latent dimensions for generator. must be same as generator latent_dim use when training GAN.

    Returns
    ---
    GAALTrainer Class

    Attributes
    ---
    generator: instance of Keras.Model of generator
    learner_acc_history: SVC accuracy history
    n_samples: number of samples in labelled set at each classifer update
    oracle: instance of Keras.Model of oracle
    start_samples: no. of starting samples
    threshold: threshold used for selecting good samples
    x_test: X of test set
    x_train: starting set of start_samples no. of training samples
    x_train_end: ending set of training samples for SVC, include generated fake samples
    y_test: y of test set
    y_train: starting set of the labesl for start_samples no. of training samples
    y_train_end: ending set of labels of training samples for SVC, include labels for generated fake samples
    '''
    def __init__(self, traindatasettype, testdatasettype, generatorpath, oraclepath, n_samples_end, threshold, start_samples, latent_dim):
        self.traindatasettype = traindatasettype
        self.testdatasettype = testdatasettype
        self.latent_dim = latent_dim

        if self.traindatasettype == 'mnist' and self.testdatasettype == 'mnist':
            print(f'Training on mnist, testing on mnist')
        elif self.traindatasettype == 'mnist' and self.testdatasettype == 'usps':
            print(f'Training on mnist, testing on usps')
        elif self.traindatasettype == 'cifar10' and self.testdatasettype == 'cifar10':
            print(f'Training and testing with cifar10')
        else:
            raise Exception('Train and test combinations allowed are: 1) mnist & mnist, 2) mnist & usps, 3) cifar10 & cifar10')

        self.threshold = threshold
        self.start_samples = start_samples
        self.x_train, self.y_train, self.x_test, self.y_test = self.loaddata()
        self.generator = self.loadgenerator(generatorpath)
        self.oracle = self.loadoracle(oraclepath)
        self.x_train_end, self.y_train_end, self.learner_acc_history, self.n_samples = self.traingaal(n_samples_end, self.x_train, self.y_train, self.x_test, self.y_test)



    def loaddata(self):
        if self.traindatasettype == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = (x_train[(y_train == 5) | (y_train == 7),]- 127.5) / 127.5
            x_train = np.expand_dims(x_train, axis=3)
            y_train = y_train[(y_train == 5) | (y_train == 7)]
            y_train = np.where(y_train == 5, 0, 1).reshape((-1,1))

            if self.testdatasettype == 'mnist':
                x_test = (x_test[(y_test == 5) | (y_test == 7),]- 127.5) / 127.5
                x_test = np.expand_dims(x_test, axis=3)
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

                y_test = y_test[(y_test == 5) | (y_test == 7)]
                y_test = np.where(y_test == 5, 0, 1).reshape((-1,1))
        
        elif self.traindatasettype == 'cifar10' and self.testdatasettype == 'cifar10':
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
        x_train = x_train[rand_idx]
        y_train = y_train[rand_idx]

        return x_train, y_train, x_test, y_test

    def trainsvc(self, xtrain, ytrain):
        x = np.squeeze(xtrain)
        xt = x.reshape((x.shape[0],-1))
        svc = SVC(kernel='linear', gamma=0.001).fit(xt, ytrain.ravel())
        return svc
    
    def loadgenerator(self, generatorpath):
        generator = load_model(generatorpath)
        generator.trainable = False
        return generator

    def loadoracle(self, oraclepath):
        oracle = load_model(oraclepath)
        oracle.trainable = False
        return oracle

    def traingaal(self, n_samples_end, x_train, y_train, x_test, y_test):
    
        x_test = np.squeeze(x_test)
        x_test = x_test.reshape((x_test.shape[0],-1))
        
        y_test = y_test.ravel()

        learner_acc = []
        n_samples = []

        n_samples.append(x_train.shape[0])

        n_samples_this = x_train.shape[0]

        svc = self.trainsvc(x_train, y_train)
        svc_preds = svc.predict(x_test)
        acc = accuracy_score(y_test, svc_preds)
        print(f'Classifier Accuracy Start: {acc}')
        learner_acc.append(acc)
        
        while n_samples_this < n_samples_end:
            # train an svc using the training set
            
            w = svc.coef_
            b = svc.intercept_

            w = tf.cast(tf.reshape(tf.Variable(w), [-1]), tf.float32)
            b = tf.cast(tf.Variable(b), tf.float32)
            
            print(f'Generating latents and adding new samples to labelled set.')
            cnt = 0
            # perform gradient descent on SVC with random start and attempt to add 10 new samples to labelled set
            while cnt < 10:

                z = np.random.randn(self.latent_dim).reshape((1,-1))
                z = tf.Variable(z)

                for _ in range(100):
                    with tf.GradientTape() as tape:
                        # Forward pass
                        tape.watch(z)
                        f = self.generator(z)
                        f = tf.reshape(f, [-1])
                        dot = tf.tensordot(f, w, 1) + b    
                        loss = (dot**2)

                    # Calculate gradients with respect to every trainable variable
                    grad = tape.gradient(loss, z)
                    opt = tf.keras.optimizers.SGD()
                    opt.apply_gradients(zip([grad], [z]))

                zt = z.numpy()
                gen = self.generator.predict(zt)

                # use oracle to predict the label of the generated picture
                if self.traindatasettype == 'mnist':
                    gen_3c = np.repeat(gen, 3, axis = 3)
                    gen_3c_32 = (tf.image.resize(gen_3c, [32,32]) + 1)/2
                    oracle_labels = np.squeeze(self.oracle.predict(gen_3c_32))
                elif self.traindatasettype == 'cifar10':
                    oracle_labels = np.squeeze(self.oracle.predict((gen + 1)/2))
                
                # pick only the good generated images subject to threshold and add to training set
                
                # append the good data to the labeled set
                if (oracle_labels < self.threshold) or (oracle_labels > (1-self.threshold)):
                    gen_good_labels = np.round(oracle_labels).astype(int)
                    gen_good_labels = np.reshape(gen_good_labels,(1,1))
                    x_train = np.concatenate((x_train, gen), axis = 0)
                    y_train = np.concatenate((y_train, gen_good_labels), axis = 0)
                    print(f'added {cnt+1}')
                    cnt += 1

            n_samples_this = x_train.shape[0]
            print(f'No. of samples in training set: {n_samples_this}')

            n_samples.append(n_samples_this)

            # Update Learner
            print(f'10 new samples added, updating SVC Classifier')
            svc = self.trainsvc(x_train, y_train)
            svc_preds = svc.predict(x_test)
            acc = accuracy_score(y_test, svc_preds)
            print(f'Classifier Accuracy: {acc}\n')
            learner_acc.append(acc)
        
        return x_train, y_train, learner_acc, n_samples