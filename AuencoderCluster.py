import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets import get_mnist_dataset, get_data_loader
from utils import *
from models import *
trainset, testset = get_mnist_dataset()
trainloader, testloader = get_data_loader(trainset, testset)
batch, labels = next(iter(trainloader))
plot_batch(batch)

class Encoder(nn.Module):
    
    def __init__(self, k):
        super().__init__()
#         self.fc1 = nn.Linear(784, 1000)
#         self.fc2 = nn.Linear(1000, 250)
        self.fc2 = nn.Linear(784, 250)
        self.fc3 = nn.Linear(250, 50)
        self.fc4 = nn.Linear(50, k)
    
    def forward(self, x):
        out = x.view(x.size(0), 784)
#         out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        return out

class Decoder(nn.Module):
    
    def __init__(self, k):
        super().__init__()
        self.fc1 = nn.Linear(k, 50)
        self.fc2 = nn.Linear(50, 250)
        self.fc3 = nn.Linear(250, 784)
#         self.fc3 = nn.Linear(250, 1000)
#         self.fc4 = nn.Linear(1000, 784)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
#         out = F.relu(self.fc4(out))
        out = out.view(out.size(0), 1, 28, 28)
        return out

class KMeansCriterion(nn.Module):
    
    def __init__(self, lmbda):
        super().__init__()
        self.lmbda = lmbda
    
    def forward(self, embeddings, centroids):
        distances = torch.sum((embeddings[:, None, :] - centroids)**2, 2)
        cluster_distances, cluster_assignments = distances.max(1)
        loss = self.lmbda * cluster_distances.sum()
        return loss, cluster_assignments


def centroid_init(k, d):
    centroid_sums = Variable(torch.zeros(k, d))
    centroid_counts = Variable(torch.zeros(k))
    for X, y in trainloader:
        X_var, y_var = Variable(X), Variable(y)
        cluster_assignments = Variable(torch.LongTensor(X.size(0)).random_(k))
        embeddings = encoder(X_var)
        update_clusters(centroid_sums, centroid_counts,
                        cluster_assignments, embeddings)
    
    centroid_means = centroid_sums / centroid_counts[:, None]
    return centroid_means.clone()

def update_clusters(centroid_sums, centroid_counts,
                    cluster_assignments, embeddings):
    k = centroid_sums.size(0)
    centroid_sums.index_add_(0, cluster_assignments, embeddings)
    np_counts = np.bincount(cluster_assignments.data.numpy(), minlength=k)
    centroid_counts.add_(Variable(torch.FloatTensor(np_counts)))

def pretrain(autoencoder, optimizer,
             print_every=100, verbose=False):
    for i, (X, y) in enumerate(trainloader):
        X_var, y_var = Variable(X), Variable(y)
        X_hat = autoencoder(X_var)
        loss = F.mse_loss(X_hat, X_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if verbose and i % print_every == 0:
            batch_hat = autoencoder(Variable(batch))
            plot_batch(batch_hat.data)
            print('Trn Loss: %.3f' % loss.data[0])

def train(encoder, decoder, centroids, optimizer, criterion,
          print_every=100, verbose=False):
    k, d = centroids.size()
    centroid_sums = torch.zeros_like(centroids)
    centroid_counts = Variable(torch.zeros(k))
    
    # run one epoch of gradient descent on autoencoders wrt centroids
    for i, (X, y) in enumerate(trainloader):
        
        # forward pass and compute loss
        X_var, y_var = Variable(X), Variable(y)
        embeddings = encoder(X_var)
        X_hat = decoder(embeddings)
        recon_loss = F.mse_loss(X_hat, X_var)
        cluster_loss, cluster_assignments = criterion(embeddings, centroids)
        loss = recon_loss + cluster_loss
        
        # run update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # store centroid sums and counts in memory for later centering
        update_clusters(centroid_sums, centroid_counts,
                        cluster_assignments, embeddings)
        
        if verbose and i % print_every == 0:
            batch_hat = autoencoder(Variable(batch))
            plot_batch(batch_hat.data)
            losses = (loss.data[0], recon_loss.data[0], cluster_loss.data[0])
            print('Trn Loss: %.3f [Recon Loss %.3f, Cluster Loss %.3f]' % losses)
    
    # update centroids based on assignments from autoencoders
    centroid_means = centroid_sums / (centroid_counts[:, None] + 1)
    return centroid_means, centroid_counts

def evaluate(encoder, decoder, loader):
    for X, y in loader:
        X_var, y_var = Variable(X), Variable(y)
        s = encoder(X_var)
        X_hat = decoder(s)
        # do something


k, d = 10, 10
encoder = Encoder(d)
decoder = Decoder(d)
autoencoder = nn.Sequential(encoder, decoder)
optimizer = optim.Adam(autoencoder.parameters())


for _ in range(1):
    pretrain(autoencoder, optimizer, verbose=True)


criterion = KMeansCriterion(lmbda=1e-3)
centroids = Variable(centroid_init(k, d).data)
plot_batch(decoder(centroids).data)

for _ in range(3):
    centroid_means, centroid_counts = train(encoder, decoder, centroids, optimizer, criterion,
                                            verbose=True)
    print(centroid_counts.data.numpy().tolist())


plot_batch(autoencoder(Variable(batch)).data)
plot_batch(decoder(centroids).data)