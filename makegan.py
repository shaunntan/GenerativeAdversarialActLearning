import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization, ReLU
from matplotlib import pyplot as plt
import tensorflow as tf

class GANTrainer:
    '''
    Train GANs

    Parameters
    ---
    datasettype: string
        'mnist' | 'cifar10'

    latent_dim: int
        no. of latent dimensions for generator. must be same as GAAL latent_dim.

    n_epochs: int
        no. of epochs to train GAN for

    batchsize: int
        batch size within each epoch

    retries: int
        no. of retry attempts to train the GAN. if GAN fails to train after max_loss_increase_epochs, training will restart.
    
    max_loss_increase_epochs: int
        max number of epochs with increase in GAN loss before restarting training

    Returns
    ---
    GANTrainer Class

    Attributes
    ---
    batchsize: batch size used when training the GAN
    datasettype: dataset type of the GAN
    discriminator: Keras.Model instance of the trained discriminator
    discriminator_accuracy_fake: training history of accuracy of discriminator on fake images
    discriminator_accuracy_real: training history of accuracy of discriminator on real images
    discriminator_losshistory_fake: loss history of accuracy of discriminator on fake images
    discriminator_losshistory_real: loss history of accuracy of discriminator on real images
    gan: Keras.Model instance of GAN
    generator: Keras.Model instance of trained generator
    generator_losshistory: loss history of accuracy of geneartor during training
    latent_dim: latent dimensions used when training generator
    max_loss_increase_epochs: max number of epochs where generator loss increases before training is restarted
    n_epochs: number of epochs to train GAN for
    retries: number of attempts allowed for training the GAN
    savepath: folder path where trained discriminator and generator is saved
    x_test: data of testing set
    x_train: data of training set
    y_test: labels of testing set
    y_train: labels of training set

    '''
    def __init__(self, datasettype, savepath, latent_dim = 100, n_epochs=200, batchsize=256, retries = 5, max_loss_increase_epochs = 10):
        self.max_loss_increase_epochs = max_loss_increase_epochs
        self.datasettype = datasettype
        self.savepath = savepath
        self.dataset = None
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batchsize = batchsize
        self.retries = retries
        print(self.savepath)
        self.gan = None

        x_train, y_train, x_test, y_test = self.loaddata()
        print('Data Loaded\n')
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        print(f'Training GAN with\n-{latent_dim} Latent Dimensions\n-for {n_epochs} Epochs\n-with Batch Size of {batchsize}\n-up to {retries} Retries\n')
        g_model, d_model, gan_model, d_loss_real, d_loss_fake, d_acc_real, d_acc_fake, g_loss = self.train(self.x_train, self.latent_dim, self.n_epochs, self.batchsize, self.retries)

        self.generator = g_model
        self.discriminator = d_model
        self.gan = gan_model
        self.discriminator_losshistory_real = d_loss_real
        self.discriminator_losshistory_fake = d_loss_fake
        self.discriminator_accuracy_real = d_acc_real
        self.discriminator_accuracy_fake = d_acc_fake
        self.generator_losshistory = g_loss

    def loaddata(self):
        if self.datasettype == 'mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = (x_train[(y_train == 5) | (y_train == 7),]- 127.5) / 127.5
            x_train = np.expand_dims(x_train, axis=3)
            y_train = y_train[(y_train == 5) | (y_train == 7)]
            y_train = np.where(y_train == 5, 0, 1).reshape((-1,1))

            x_test = (x_test[(y_test == 5) | (y_test == 7),]- 127.5) / 127.5
            x_test = np.expand_dims(x_test, axis=3)
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

        return x_train, y_train, x_test, y_test

    def makediscriminator(self):
        model = Sequential()
        model.add(Conv2D(64, (5,5), strides=(2, 2), padding='same', input_shape = self.x_train.shape[1:]))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5,5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def makegenerator(self, latent_dim):
        if self.datasettype == 'mnist':
            channels = 1
            finalsize = 28
        elif self.datasettype == 'cifar10':
            channels = 3
            finalsize = 32

        model = Sequential()
        if self.datasettype == 'mnist':

            model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Reshape((7, 7, 256)))

            model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            model.add(BatchNormalization())
            model.add(LeakyReLU())

            model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

        elif self.datasettype == 'cifar10':
            n_nodes = 256 * 4 * 4
            model.add(Dense(n_nodes, input_dim=latent_dim))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Reshape((4, 4, 256)))
            # upsample to 8x8
            model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            # upsample to 16x16
            model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            # upsample to 32x32
            model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            # output layer
            model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
        return model

    def makegan(self, g_model, d_model):
        # set discriminator to not trainable
        d_model.trainable = False
        # setup gan
        model = Sequential()
        model.add(g_model)
        model.add(d_model)
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def gen_real(self, dataset, n_samples):
        # generate random indices for subsampling
        idx = np.random.randint(0, dataset.shape[0], n_samples)
        x = dataset[idx]
        # generate class labels == 1
        y = np.ones((n_samples, 1))
        return x, y

    def gen_fake(self, g_model, latent_dim, n_samples):
        # generate points in latent space
        x_input = self.get_latent(latent_dim, n_samples)
        # predict outputs
        X = g_model.predict(x_input)
        # create 'fake' class labels (0)
        y = np.zeros((n_samples, 1))
        return X, y

    def get_latent(self, latent_dim, n_samples):
        # generate latent input points
        x_input = np.random.randn(latent_dim * n_samples)
        # reshape to fit network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    def save_plot(self, examples, epoch, n=4):
        # plot images
        print(examples.shape)
        print(np.min(examples))
        print(np.max(examples))
        for i in range(n * n):
            # define subplot
            plt.subplot(n, n, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(examples[i].astype('uint8'))
        # save plot to file
        filename = 'Generated_plot_e%03d.png' % (epoch+1)
        plt.savefig(self.savepath + filename)
        plt.close()

    def performance(self, epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
        # prepare real samples
        X_real, y_real = self.gen_real(dataset, n_samples)
        # evaluate discriminator on real examples
        _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.gen_fake(g_model, latent_dim, n_samples)
        # evaluate discriminator on fake examples
        _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print('Performance test> Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
        # save plot
        fakeplot = (x_fake * 127.5) + 127.5

        self.save_plot(fakeplot, epoch)
        # save the generator model tile file
        g_filename = 'generator_model_%03d.h5' % (epoch + 1)
        g_model.save(self.savepath + g_filename)
        d_filename = 'discriminator_model_%03d.h5' % (epoch + 1)
        d_model.save(self.savepath + d_filename)

    def train(self, dataset, latent_dim, n_epochs=200, batchsize=256, retries = 5):
        batch_per_epoch = int(dataset.shape[0] / batchsize)
        half_batch = int(batchsize / 2)
        # manually enumerate epochs

        for r in range(retries):
            print(f'Attempt:{r+1}')
            d_model = self.makediscriminator()
            g_model = self.makegenerator(latent_dim)
            gan_model = self.makegan(g_model, d_model)
            
            d_loss_real = []
            d_loss_fake = []
            d_acc_real = []
            d_acc_fake = []

            g_loss = []
            g_loss_epoch = []
            g_loss_inc_counter = 0
            for i in range(n_epochs):
                print(f'Epoch: {i+1}')
                # enumerate batches over the training set
                for j in range(batch_per_epoch):
                    # get randomly selected 'real' samples
                    X_real, y_real = self.gen_real(self.x_train, half_batch)
                    # generate 'fake' examples
                    X_fake, y_fake = self.gen_fake(g_model, self.latent_dim, half_batch)
                    # update discriminator model weights
                    d_l_real, d_a_real = d_model.train_on_batch(X_real, y_real)
                    d_l_fake, d_a_fake = d_model.train_on_batch(X_fake, y_fake)
                    # prepare points in latent space as input for the generator
                    X_gan = self.get_latent(self.latent_dim, self.batchsize)
                    # create inverted labels for the fake samples
                    y_gan = np.ones((self.batchsize, 1))
                    # update the generator via the discriminator's error
                    g_l = gan_model.train_on_batch(X_gan, y_gan)
                    # summarize loss on this batch
                    d_loss_real.append(d_l_real)
                    d_loss_fake.append(d_l_fake)
                    d_acc_real.append(d_a_real)
                    d_acc_fake.append(d_a_fake)
                    g_loss.append(g_l)
                print('Try:%d, Epoch:%d, D_Loss_Real=%.3f, D_Loss_Fake=%.3f, D_Acc_Real=%.3f, D_Acc_Fake=%.3f, GAN_Loss=%.3f\n' % (r+1, i+1, d_l_real, d_l_fake, d_a_real, d_a_fake, g_l))
                g_loss_epoch.append(g_l)
                if len(g_loss_epoch) >= 2:
                    if g_loss_epoch[-1] >  g_loss_epoch[-2]:
                        g_loss_inc_counter += 1
                        print(f'Generator loss increased for {g_loss_inc_counter} epochs.')
                    elif g_loss_epoch[-1] <  g_loss_epoch[-2]:
                        g_loss_inc_counter = 0

                if g_loss_inc_counter == self.max_loss_increase_epochs:
                    print(f'Try: {r+1} failed to train. Restarting training')
                    break
        #         evaluate the model performance, sometimes
                if (i+1) % 20 == 0:
                    self.performance(i, g_model, d_model, dataset, latent_dim)
                
                if i == self.n_epochs - 1:
                    return g_model, d_model, gan_model, d_loss_real, d_loss_fake, d_acc_real, d_acc_fake, g_loss

        # reach here if fail to train
        print("Training Stopped...")
        return g_model, d_model, gan_model, d_loss_real, d_loss_fake, d_acc_real, d_acc_fake, g_loss
        