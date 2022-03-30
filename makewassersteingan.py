import numpy as np
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization, ReLU
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm

class WGANTrainer:
    '''
    Train GANs using Wasserstein Loss method aka Wasserstein GAN

    Parameters
    ---
    datasettype: string
        'mnist' | 'cifar10'

    latent_dim: int
        no. of latent dimensions for generator. must be same as GAAL latent_dim.

    n_epochs: int
        no. of epochs to train GAN for
    
    n_critic: int
        no. of critic updates per batch (WGAN)

    batchsize: int
        batch size within each epoch

    retries: int
        no. of retry attempts to train the GAN. if GAN fails to train after max_loss_increase_epochs, training will restart.
    
    Returns
    ---
    WGANTrainer Class

    Attributes
    ---
    batchsize: batch size used when training the GAN
    c_losses: losses from discriminator training
    c_model: Keras.Model instance of the trained critic
    datasettype: dataset type of the GAN
    g_losses: losses from generator training
    g_model: Keras.Model instance of the trained generator
    latent_dim: latent dimensions used when training generator
    n_critic: number of times critic is updated in each batch (Wasserstein GAN)
    n_epochs: number of epochs to train GAN for
    retries: number of attempts allowed for training the GAN
    savepath: folder path where trained discriminator and generator is saved
    x_test: data of testing set
    x_train: data of training set
    y_test: labels of testing set
    y_train: labels of training set

    '''
    def __init__(self, datasettype, savepath, latent_dim = 100, n_epochs=200, batchsize=256, retries = 5, n_critic = 5):
        self.datasettype = datasettype
        self.savepath = savepath
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batchsize = batchsize
        self.retries = retries
        print(self.savepath)
        self.n_critic = n_critic

        self.x_train, self.y_train, self.x_test, self.y_test = self.loaddata()
        print('Data Loaded\n')


        print(f'Training Wasserstein GAN with\n-{latent_dim} Latent Dimensions\n-for {n_epochs} Epochs\n-with Batch Size of {batchsize}\n-up to {retries} Retries\n')
        self.g_model, self.c_model, self.c_losses, self.g_losses = self.train(self.x_train, self.latent_dim, self.n_epochs, self.batchsize, self.retries, self.n_critic)

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

    def makecritic(self):
        model = Sequential()
        model.add(Conv2D(64, (5,5), strides=(2, 2), padding='same', input_shape = self.x_train.shape[1:]))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5,5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.3))

        model.add(Flatten())
        ### WGAN-Change to linear activation
        model.add(Dense(1, activation='linear'))
        ### optimisation using GradientTape method
        # opt = Adam(lr=0.0002, beta_1=0.5)
        # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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

    def gen_real(self, dataset, n_samples):
        # generate random indices for subsampling
        idx = np.random.randint(0, dataset.shape[0], n_samples)
        x = dataset[idx]
        ### for WGAN generate class labels == -1 for real
        y = -np.ones((n_samples, 1))
        return x, y

    def gen_fake(self, g_model, latent_dim, n_samples):
        # generate points in latent space
        x_input = self.get_latent(latent_dim, n_samples)
        # predict outputs
        X = g_model.predict(x_input)
        ### for WGAN generate class labels == 1 for fake
        y = np.ones((n_samples, 1))
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

    def performance(self, epoch, g_model, c_model, dataset, latent_dim, n_samples=100):

        # prepare fake examples
        x_fake, _ = self.gen_fake(g_model, latent_dim, n_samples)
        # save plot
        fakeplot = (x_fake * 127.5) + 127.5

        self.save_plot(fakeplot, epoch)
        # save the generator model tile file
        g_filename = 'generator_model_%03d.h5' % (epoch + 1)
        g_model.save(self.savepath + g_filename)
        d_filename = 'discriminator_model_%03d.h5' % (epoch + 1)
        c_model.save(self.savepath + d_filename)

    def generator_loss(self, fake_output):
        gen_loss = -1. * tf.math.reduce_mean(fake_output)
        return gen_loss

    def discriminator_loss(self, real_output, fake_output):
        loss = -1. * (tf.math.reduce_mean(real_output) - tf.math.reduce_mean(fake_output))
        return loss

    def train(self, dataset, latent_dim, n_epochs=200, batchsize=256, retries = 5, n_critic = 5):
        batch_per_epoch = int(dataset.shape[0] / batchsize)
        # manually enumerate epochs
        
        for r in range(retries):
            print(f'Attempt:{r+1}')
            c_model = self.makecritic()
            g_model = self.makegenerator(latent_dim)

            c_losses = []
            g_losses = []
            g_loss_epoch = []

            opt_critic = RMSprop(learning_rate=0.00005)
            opt_generator = RMSprop(learning_rate=0.00005)

            for i in range(n_epochs):
                print(f'Epoch: {i+1}')
                for j in tqdm(range(batch_per_epoch)):
                    ### Update Critic more than Generator
                    for _ in range(n_critic):
                        # Get randomly selected 'real' samples
                        X_real, _ = self.gen_real(dataset, batchsize)
                        # Generate 'fake' examples
                        X_fake, _ = self.gen_fake(g_model, latent_dim, batchsize)

                        with tf.GradientTape() as tape:
                            # tape.watch(c_model.trainable_variables)
                            c_real_output = c_model(X_real, training = True)
                            c_fake_output = c_model(X_fake, training = True)

                            c_loss = self.discriminator_loss(c_real_output, c_fake_output)

                        gradients_c_model = tape.gradient(c_loss, c_model.trainable_variables)

                        opt_critic.apply_gradients(zip(gradients_c_model, c_model.trainable_variables))

                    # Clip weights of Critic
                    for w in c_model.trainable_variables:
                        w.assign(tf.clip_by_value(w, -0.01, 0.01))

                    # Prepare points in latent space as input for the generator
                    latents = self.get_latent(self.latent_dim, self.batchsize)

                    ### Update Generator
                    with tf.GradientTape() as gen_tape:
                        # tape.watch(g_model.trainable_variables)
                        generated_images = g_model(latents, training=True)
                        g_fake_output = c_model(generated_images, training=False)
                        g_loss = self.generator_loss(g_fake_output)
                    
                    gradients_g_model = gen_tape.gradient(g_loss, g_model.trainable_variables)

                    opt_generator.apply_gradients(zip(gradients_g_model, g_model.trainable_variables))

                    # summarize loss on this batch
                    g_losses.append(g_loss)
                    c_losses.append(c_loss)

                print('Try:%d, Epoch:%d, C_Loss=%.3f, G_Loss=%.3f\n' % (r+1, i+1, c_loss, g_loss))
                g_loss_epoch.append(g_loss)

        #         evaluate the model performance, sometimes
                if (i+1) % 20 == 0:
                    self.performance(i, g_model, c_model, dataset, latent_dim)
                
                if i == self.n_epochs - 1:
                    return g_model, c_model, c_losses, g_losses

        # reach here if fail to train
        print("Training Stopped...")
        return g_model, c_model, c_losses, g_losses
        