from __future__ import division
import argparse, sys
import imp
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import load_mnist, get_speech

def get_sample(model, Y):
    gmm = model[Y]

    sample = gmm.sample()
    mean = gmm.means_[sample[1]]

    return np.reshape(sample[0], newshape=(28, 28)), np.reshape(mean, newshape=(28, 28))

def gmm_fit(X, Y):
    K = len(set(Y))

    gaussian = []
    for k in range(K):
        Xk = X[Y == k]
        
        gmm = BayesianGaussianMixture(20, init_params='random')
        gmm.fit(Xk)

        gaussian.append(gmm)

    return gaussian


def main(params):
    X_train, Y_train, _, _ = load_mnist()

    # Training the simple Gaussian model
    model = gmm_fit(X_train, Y_train)

    # Plot the results from the train model and compare with mean training values
    # Input values here is class label Y (Y in range(0, 10))
    _, ax = plt.subplots(nrows=5, ncols=4)

    for k in range(len(set(Y_train))):
        # Getting the sample from the trained model
        sample, mean = get_sample(model, k)

        ax[k//2, (k%2)*2].imshow(mean, cmap='gray')
        ax[k//2, (k%2)*2].set_title('mean of {}'.format(k))
        ax[k//2, (k%2)*2].axis('off')
        ax[k//2, (k%2)*2+1].imshow(sample, cmap='gray')
        ax[k//2, (k%2)*2+1].set_title('generative sample of {}'.format(k))
        ax[k//2, (k%2)*2+1].axis('off')
        
    plt.show()

    if params.useSpeechRecognition:
        while True:
            number = get_speech()

            if number == 'stop':
                print('DONE, STOP THE PROGRAM!')
                break
            elif number != 'no':
                sample, mean = get_sample(model, number)

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
            sample, _ = get_sample(model, number)

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