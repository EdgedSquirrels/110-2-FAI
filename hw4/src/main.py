from operator import mod
import random, os
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from autoencoder import Autoencoder, DenoisingAutoencoder
from pca import PCA

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def plot_component(image, output_path):
    plt.imshow(image.reshape(61, 80), cmap='gray')
    plt.savefig(output_path)

def plot_2component(image1, image2, output_path):
    plt.autoscale(True)
    plt.subplot(1,2,1)
    plt.title('original')
    plt.imshow(image1.reshape(61, 80), cmap='gray')
    plt.subplot(1,2,2)
    plt.title('reconstruct')
    plt.imshow(image2.reshape(61, 80), cmap='gray')
    plt.savefig(output_path)

    

def sample_architecture(self, latent_space_dim):
    self.encoder = nn.Sequential(
        nn.Linear(80 * 61, 768),
        nn.ReLU(),
        nn.Linear(768, 128),
        nn.ReLU(),
        nn.Linear(128, latent_space_dim),
        nn.ReLU()
    )
    self.decoder = nn.Sequential(
        nn.Linear(latent_space_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 768),
        nn.ReLU(),
        nn.Linear(768, 80 * 61),
        nn.Sigmoid()
    )

def architecture2(self, latent_space_dim):
    self.encoder = nn.Sequential(
        nn.Linear(80 * 61, 2048),
        nn.ReLU(),
        nn.Linear(2048, 768),
        nn.ReLU(),
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, latent_space_dim),
        nn.ReLU()
    )
    self.decoder = nn.Sequential(
        nn.Linear(latent_space_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, 768),
        nn.ReLU(),
        nn.Linear(768, 2048),
        nn.ReLU(),
        nn.Linear(2048, 80 * 61),
        nn.Sigmoid()
    )

def architecture3(self, latent_space_dim):
    self.encoder = nn.Sequential(
        nn.Linear(80 * 61, 256),
        nn.ReLU(),
        nn.Linear(256, latent_space_dim),
        nn.ReLU()
    )
    self.decoder = nn.Sequential(
        nn.Linear(latent_space_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 80 * 61),
        nn.Sigmoid()
    )

def main(args):
    set_seed(args.seed)
    """
    Load dataset
    Dataset: stores the samples and their corresponding labels
    Dataloader: wraps an iterable around the Dataset to enable easy access to the samples.
    """
    SPLIT = ('train', 'val')
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x) / 255)
    ])
    dataset = {x: ImageFolder(os.path.join(args.data_dir, x), transform=transform) for x in SPLIT}
    dataloader = {x: DataLoader(dataset[x], batch_size=len(dataset[x])) for x in SPLIT}

    X_train, y_train = next(iter(dataloader['train']))
    X_val, y_val = next(iter(dataloader['val']))
   
    pca = PCA(args)
    autoencoder = Autoencoder(args, sample_architecture)
    denoising_autoencoder = DenoisingAutoencoder(args, sample_architecture)
    '''
    architecture2
    mean squared error: 0.011790511
    0.6

    architecture3
    mean squared error: 0.008102858
    0.5333333333333333

    pca
    mean squared error: 0.0023547083
    0.9

    autoencoder
    mean squared error: 0.0029266474
    0.7333333333333333

    denoisingautoencoder
    mean squared error: 0.0017002755
    0.7




    '''

    logistic_regression = LogisticRegression(solver='liblinear', max_iter=500)
    
    # TODO hw4
    # for model in [denoising_autoencoder]:
    for model in [pca, autoencoder, denoising_autoencoder]:
        # encode the images into a lower dimensional latent representation with PCA and autoencoders
        X_train_transformed = model.fit_transform(X_train)
        reconstructed_image = model.reconstruct(X_train_transformed[41])
        if isinstance(model, PCA):
            plot_component(model.Xmean, "eigen/meanface")
            for i in range(model.n_components):
                plot_component(model.transform_matrix[i], "eigen/eigenface%d" % (i))
        """
        TODO: Plot the reconstructed images.
              Calculate the mean squared error between the reconstructed image and the original image.
        """
        # print(reconstructed_image.shape)
        # print(reconstructed_image)
        if isinstance(model, PCA):
            plot_2component(X_train[41], reconstructed_image, "reconstructPCA")
        elif isinstance(model, DenoisingAutoencoder):
            plot_2component(X_train[41], reconstructed_image, "reconstructDAE")
        else:
            plot_2component(X_train[41], reconstructed_image, "reconstructAE")

        print("mean squared error:", np.average((X_train[41] - reconstructed_image) ** 2))

        # train and predict with logistic regression
        logistic_regression.fit(X_train_transformed, y_train)
        X_val_transformed = model.transform(X_val)
        y_pred = logistic_regression.predict(X_val_transformed)
        print(accuracy_score(y_val, y_pred))

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_space_dim', type=int, default=16)
    parser.add_argument('--noise_factor', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

if __name__ == '__main__':
    main(parse_arguments())