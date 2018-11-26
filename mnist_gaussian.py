from __future__ import division
import argparse, sys
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import load_mnist, get_speech

def gaussian_fit(X, Y):
    ''' Calculate mean and covariance of images from the training data
    '''
    K = len(set(Y))

    gaussian = []
    for k in range(K):
        Xk = X[Y == k]
        mu = Xk.mean(axis=0)
        cov = np.cov(Xk.T)
        g = {'mu': mu, 'cov': cov}
        gaussian.append(g)

    return gaussian

def get_sample(model, Y):
    return np.reshape( mvn.rvs(mean=model[Y]['mu'], cov=model[Y]['cov']), newshape=(28,28) )


def main(params):
    X_train, Y_train, _, _ = load_mnist()

    # Training the simple Gaussian model
    model = gaussian_fit(X_train, Y_train)

    # Plot the results from the train model and compare with mean training values
    # Input values here is class label Y (Y in range(0, 10))
    _, ax = plt.subplots(nrows=5, ncols=4)

    for k in range(len(set(Y_train))):
        # Getting the sample from the trained model
        sample = get_sample(model, k)
        mean = np.reshape(model[k]['mu'], newshape=(28, 28))

        ax[k//2, (k%2)*2].imshow(mean, cmap='gray')
        ax[k//2, (k%2)*2].set_title('mean of {}'.format(k))
        ax[k//2, (k%2)*2].axis('off')
        ax[k//2, (k%2)*2+1].imshow(sample, cmap='gray')
        ax[k//2, (k%2)*2+1].set_title('generative sample of {}'.format(k))
        ax[k//2, (k%2)*2+1].axis('off')
        
    plt.show()

    print('YES, still running, now test the speech recognition!!')

    if params.useSpeechRecognition:
        while True:
            number = get_speech()

            if number == 'stop':
                print('DONE, STOP THE PROGRAM!')
                break
            elif number != 'no':
                sample = get_sample(model, number)
                mean = np.reshape(model[number]['mu'], newshape=(28, 28))

                fig, ax = plt.subplots(nrows=1, ncols=2)
                ax[0].imshow(mean, cmap='gray')
                ax[0].set_title('Mean')
                ax[1].imshow(sample, cmap='gray')
                ax[1].set_title('generative sample of {}'.format(9))
                fig.tight_layout()
                plt.show()
            else:
                print('CAN NOT GENERATE!')

    else:
        plt.figure(figsize = (8, 8))
        gs1 = gridspec.GridSpec(8, 8)
        gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 

        for i in range(64):
            number = np.random.randint(low=0, high=10)
            # Getting the sample from the trained model
            sample = get_sample(model, number)

            ax1 = plt.subplot(gs1[i])
            plt.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            ax1.imshow(sample, cmap='gray')

        plt.show()


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Fully supervised monocular depth with ResNet.')
    parser.add_argument('-useSpeech', '--useSpeechRecognition', dest='useSpeechRecognition',
                        help='Flag enable the use of speech recognition rather than typing.',
                        action='store_true')
    # parser.add_argument('-m', '--usingModel', dest='usingModel',
    #                     help='The model using to train the data.', required=True)
    args = parser.parse_args()
    
    main(args)