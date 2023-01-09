from data_transformer import DataTransformer
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt



'''
def forward(self, inputs):
    codes = self.encoder(inputs)
    decoded = self.decoder(codes)
    return codes, decoded
'''

class AutoencoderModel(nn.Module):
    """Add more of your code here if you want to"""
    def __init__(self, latent_space_dim, architecture_callback):
        super().__init__()
        architecture_callback(self, latent_space_dim)
        # sample_architecture3(self, latent_space_dim)

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded



class Autoencoder(DataTransformer):
    """Add more of your code here if you want to"""
    def __init__(self, args, architecture_callback):
        self.args = args
        self.architecture_callback = architecture_callback
        # self.model = AutoencoderModel(args.latent_space_dim, architecture_callback).to(args.device)

    def fit(self, X):
        epochs = self.args.num_epochs
        batch_size = self.args.batch_size
        lr = self.args.learning_rate
        device = self.args.device
        train_loader = DataLoader(X, batch_size=batch_size, shuffle=True)
        model_ae = self.model = AutoencoderModel(self.args.latent_space_dim, self.architecture_callback).to(device)
        optimizer = torch.optim.Adam(model_ae.parameters(), lr=lr)
        loss_function = nn.MSELoss().to(device)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,40], gamma=0.5)

        # Train
        log_loss = []
        plt.plot([.9435, .29135845, .20385234, .23545,.04375])

        for epoch in range(epochs):
            total_loss = 0
            for data in train_loader:
                inputs = data.view(-1, 80 * 61).to(device)
                # print(data.shape, inputs.shape)
                # print(data)
                # print(inputs)
                model_ae.zero_grad()
                # Forward
                codes, decoded = model_ae(inputs)
                loss = loss_function(decoded, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss
                log_loss.append(loss.item())
            total_loss /= len(train_loader.dataset)
            scheduler.step()

            if epoch % 5 == 0:
                print('[{}/{}] Loss:'.format(epoch+1, epochs), total_loss.item())
        print('[{}/{}] Loss:'.format(epoch+1, epochs), total_loss.item())

        # print(log_loss)
        plt.subplot(1, 1, 1)
        plt.plot(log_loss)
        plt.savefig("AEloss")
        torch.save(model_ae, 'mode_AutoEncoder_MNIST.pth')

        # optimize restruction error
        
        # raise NotImplementedError
    
    def transform(self, X):

        '''
        model_ae = torch.load('mode_AutoEncoder_MNIST.pth')
        model_ae.eval()
        train_loader = DataLoader(X, batch_size=self.args.batch_size)
        with torch.no_grad():
            codes = model_ae()
        '''
        x = self.model.encoder(X)
        x = x.detach()
        # print(x)
        return x
        # raise NotImplementedError
    
    def reconstruct(self, X_transformed):
        x = self.model.decoder(X_transformed)
        # x.backward()
        # print(x)
        x = x.detach()
        # print(x)
        return x
        # raise NotImplementedError


class DenoisingAutoencoder(Autoencoder):
    """Add more of your code here if you want to"""
    def __init__(self, args, architecture_callback):
        super().__init__(args, architecture_callback)
    
    def fit(self, X):
        
        epochs = self.args.num_epochs
        batch_size = self.args.batch_size
        lr = self.args.learning_rate
        device = self.args.device

        train_loader = DataLoader(X, batch_size=batch_size, shuffle=True)
        model_ae = self.model = AutoencoderModel(self.args.latent_space_dim, self.architecture_callback).to(device)
        optimizer = torch.optim.Adam(model_ae.parameters(), lr=lr)
        loss_function = nn.MSELoss().to(device)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,40], gamma=0.5)


        # Train
        log_loss = []
        for epoch in range(epochs):
            total_loss = 0
            for data in train_loader:
                inputs = data.view(-1, 80 * 61).to(device)
                epsilon = torch.normal(0, self.args.noise_factor, size=inputs.shape)
                model_ae.zero_grad()

                # Forward
                codes, decoded = model_ae(inputs + epsilon)
                loss = loss_function(decoded, inputs)
                loss.backward()
                optimizer.step()
                total_loss += loss
                log_loss.append(loss.item())
            total_loss /= len(train_loader.dataset)
            scheduler.step()

            if epoch % 5 == 0:
                print('[{}/{}] Loss:'.format(epoch+1, epochs), total_loss.item())
        print('[{}/{}] Loss:'.format(epoch+1, epochs), total_loss.item())

        # print(log_loss)
        plt.subplot(1, 1, 1)
        plt.plot(log_loss)
        plt.savefig("DAEloss")
        torch.save(model_ae, 'mode_DenoisingAutoEncoder_MNIST.pth')