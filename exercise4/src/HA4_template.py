# ## 4. Dimensionality Reduction, Representation Learning & Generative Modelling

#%%

# Import Pytorch and other relevant packages
import torch
import torch.nn as nn
# Import MNIST dataset 
from torchvision.datasets import MNIST
# Load Numpy and Matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


#%%

# 1. Download MNIST data
train_set = MNIST('.', train=True, download=True)
test_set = MNIST('.', train=False, download=True)


#%%

# 2. Check number of data points

print(train_set.data.shape)
print(train_set.targets.shape)

print(test_set.data.shape)
print(test_set.targets.shape)


#%%

# 3. Histogram

label_bincount = torch.bincount(train_set.targets).numpy()
label_bins = list(range(len(label_bincount)))

ax = sns.barplot(x=label_bins, y=label_bincount)
ax.set_xlabel("label")
ax.set_ylabel("frequency")
plt.show()


#%%

# 4. Visualize

fig, axs = plt.subplots(2, 5)

i = 0

for row in axs:
    for ax in row:
        ax.imshow(train_set.data[i], cmap="Greys")
        i += 1
        ax.axis('off')

fig.tight_layout()
plt.show()


#%%

# 5. Reshape the images from Nx28x28 to Nx784

# Also transform to [0, 1]
X_train = torch.flatten(train_set.data, 1).float() / 255
X_test = torch.flatten(test_set.data, 1).float() / 255

print(X_train.shape)
print(X_test.shape)


#%%

# # 4.2 Principal Component Analysis

#%%

# 1. Subset of data with [0,1,2,3,4] classes only

train_reduced_index = (train_set.targets < 5)
X_train_reduced = X_train[train_reduced_index]
y_train_reduced = train_set.targets[train_reduced_index]

test_reduced_index = (test_set.targets < 5)
X_test_reduced = X_test[test_reduced_index]
y_test_reduced = test_set.targets[test_reduced_index]


#%%

# 2. PCA implementation from scikit-learn
from sklearn.decomposition import PCA


#%%

# 3. Perform PCA

pca = PCA(n_components=200)
pca.fit(X_train_reduced)

#%%

# 4. Plot the Eigen spectrum

ax = sns.barplot(x=list(range(len(pca.explained_variance_ratio_))), y=pca.explained_variance_ratio_)
ax.set_xlabel("component")
ax.set_ylabel("fraction of variance explained")

plt.xticks(range(0, 200, 20))
plt.show()

print(f"Total explained variance: {sum(pca.explained_variance_ratio_) * 100:.2f}%")

#%%

# 5. Vary number of principal components

pca = PCA()
pca.fit(X_train_reduced)

D = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

var_explained = [sum(pca.explained_variance_ratio_[:d]) for d in D]

fig, ax = plt.subplots()
plt.plot(D, var_explained, marker='o')
ax.set_xlabel("number of components")
ax.set_ylabel("fraction of variance explained")
plt.show()

#%%

# 6. Transform to 2D

pca = PCA(n_components=2)
X_train_transformed = pca.fit_transform(X_train_reduced)

X_test_transformed = pca.transform(X_test_reduced)


#%%

# 8. k-Means and visualization from 6. and 7.

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def clustering(X_train_transformed, y_train_reduced, X_test_transformed, y_test_reduced):
    kmeans = KMeans(n_clusters=5, random_state=2)

    y_train_pred = kmeans.fit_predict(X_train_transformed)

    mapping = [np.argmax(np.bincount(y_train_reduced[y_train_pred == i])) for i in range(5)]

    y_train_pred_mapped = np.array(mapping)[y_train_pred]

    y_test_pred = kmeans.predict(X_test_transformed[:, :2])
    y_test_pred_mapped = np.array(mapping)[y_test_pred]

    for Xs, ys in [(X_train_transformed, y_train_reduced), (X_train_transformed, y_train_pred), (X_train_transformed, y_train_pred_mapped), (X_test_transformed, y_test_reduced), (X_test_transformed, y_test_pred), (X_test_transformed, y_test_pred_mapped)]:
        ax = sns.scatterplot(x=Xs[:, 0], y=Xs[:, 1], hue=ys, marker=".", alpha=0.3, palette="bright")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()

    print("Training scores:")
    print(f"Accuracy, mapped labels: {np.mean(y_train_pred_mapped == y_train_reduced.numpy()) * 100:.2f}%")
    print(f"Adjusted random index score: {adjusted_rand_score(y_train_reduced, y_train_pred)}")
    print(f"Adjusted mutual info score: {adjusted_mutual_info_score(y_train_reduced, y_train_pred)}")
    print()
    print("Testing scores:")
    print(f"Testing accuracy, mapped labels: {np.mean(y_test_pred_mapped == y_test_reduced.numpy()) * 100:.2f}%")
    print(f"Adjusted random index score: {adjusted_rand_score(y_test_reduced, y_test_pred)}")
    print(f"Adjusted mutual info score: {adjusted_mutual_info_score(y_test_reduced, y_test_pred)}")


clustering(X_train_transformed, y_train_reduced, X_test_transformed, y_test_reduced)

#%%

# # 4.3 Autoencoders

#%%

# Import additional torch modules
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

cuda = True  # Set this if training on GPU
cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print("Using "+repr(device))


#%%

from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.y)


train_dataset = MNISTDataset(X_train_reduced, y_train_reduced)
test_dataset = MNISTDataset(X_test_reduced, y_test_reduced)


#%%

N = len(train_dataset)
nTrain = int(0.8*N)
nValid = N-nTrain


valid_set, train_dataset = random_split(train_dataset,(nValid,nTrain))
print("Using nTrain: %d, nValid: %d "%(nTrain, nValid))


#%%


# Autoencoder class
class AE(nn.Module):
    def __init__(self, latent_dim):
        super(AE, self).__init__()
        # Encoder layers
        self.fc_enc1 = nn.Linear(784, 32)
        self.fc_enc2 = nn.Linear(32, 16)
        self.fc_enc3 = nn.Linear(16, latent_dim)
        
        # Decoder layers
        self.fc_dec1 = nn.Linear(latent_dim, 16)
        self.fc_dec2 = nn.Linear(16,32)
        self.fc_dec3 = nn.Linear(32,784)
        
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = F.relu(self.fc_enc1(x))
        x = F.relu(self.fc_enc2(x))
        x = F.relu(self.fc_enc3(x))

        return x
    
    def decode(self, x):
        x = F.relu(self.fc_dec1(x))
        x = F.relu(self.fc_dec2(x))
        x = self.sigmoid(self.fc_dec3(x))
        
        return x

    def forward(self, x):
        # Autoencoder returns the reconstruction 
        # and latent representation
        z = self.encode(x)
        # decode z
        xHat = self.decode(z)
        return xHat,z


#%%

# Set parameters for the model
torch.manual_seed(42)  # set fixed random seed for reproducibility

batch_size = 2 ** 11
epochs = 500
latent_dim = 2
lr = 1e-6 * batch_size

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=cuda)
valid_loader = torch.utils.data.DataLoader(valid_set,
    batch_size=len(valid_set), shuffle=True, num_workers=4,
                              pin_memory=cuda)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=cuda)

model_ae = AE(2).to(device)
optimizer = optim.Adam(model_ae.parameters(), lr=lr)
loss_function = nn.BCELoss()


#%%


from tqdm import tqdm

valid_loss = float("inf")
val = []

with tqdm(total=epochs * nTrain, unit='images', desc='Training Model') as pbar:
    for epoch in range(1, epochs + 1):
        train_loss = 0

        model_ae.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            xHat, z = model_ae(data)

            loss = loss_function(xHat, data)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pbar.update(len(data))
            pbar.set_postfix_str(f'train loss: {loss.item():>3f}, val loss: {valid_loss:>3f}, epoch: {epoch}')

        valid_loss = 0
        model_ae.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(valid_loader):
                data = data.to(device)
                xHat, z = model_ae(data)
                valid_loss += loss_function(xHat, data).item()

                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data.view(len(data), 1, 28, 28)[:n],
                                          xHat.view(len(data), 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                             'results/reconstruction_ae_' + str(epoch) + '.png', nrow=n)

        valid_loss /= len(valid_loader)

        val.append(valid_loss)

fig, ax = plt.subplots()
ax.plot(list(range(epochs)), val, marker='o', label="val loss")
ax.legend()
ax.set_xlabel("epoch")
ax.set_ylabel("validation loss")
plt.show()


#%%

# Use converged model to evaluate on train and test set.

model_ae.eval()

with torch.no_grad():
    xHat_train, z_train = model_ae(X_train_reduced.to(device))
    xHat_train = xHat_train.cpu()
    z_train = z_train.cpu()
    train_loss = loss_function(xHat_train, X_train_reduced).item()

    xHat_test, z_test = model_ae(X_test_reduced.to(device))
    xHat_test = xHat_test.cpu()
    z_test = z_test.cpu()
    test_loss = loss_function(xHat_test, X_test_reduced).item()

print('Train loss: {:.4f}'.format(train_loss))
print('Test loss: {:.4f}'.format(test_loss))

clustering(z_train, y_train_reduced, z_test, y_test_reduced)


#%%


# Reconstruction + KL divergence losses summed over all elements and batch
def elbo_loss(recon_x, x, mu, logvar, KLD_weight):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')

    # reduction="mean"
    KLD = KLD_weight * torch.mean(- 0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), -1))

    return BCE + KLD


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        # Encoder layers
        self.fc_enc1 = nn.Linear(784, 32)
        self.fc_enc2 = nn.Linear(32, 16)
        self.fc_enc3 = nn.Linear(16, 2*latent_dim)  # Note we return 2*latent_dim

        # Decoder layers
        self.fc_dec1 = nn.Linear(latent_dim, 16)
        self.fc_dec2 = nn.Linear(16,32)
        self.fc_dec3 = nn.Linear(32,784)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        z = mu + std * eps

        return z

    def encode(self, x):
        x = F.relu(self.fc_enc1(x))
        x = F.relu(self.fc_enc2(x))
        x = self.fc_enc3(x)

        mu, logvar = torch.chunk(x, 2, -1)

        return mu, logvar


    def decode(self, z):
        x = F.relu(self.fc_dec1(z))
        x = F.relu(self.fc_dec2(x))
        x = self.sigmoid(self.fc_dec3(x))

        return x

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        xHat = self.decode(z)

        return xHat, mu, logvar


#%%

# Set parameters for the model
torch.manual_seed(42) # set fixed random seed for reproducibility

batch_size = 2 ** 12
epochs = 500
latent_dim = 2
lr = 5e-7 * batch_size

# KLD is weighted 1 / n_features to obtain same weighting using reduction="mean" as when using reduction="sum"
# Mean reduction is preferred to make loss with different batch sizes comparable
KLD_weight = 1 / X_train_reduced.shape[1]

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=cuda)
valid_loader = torch.utils.data.DataLoader(valid_set,
    batch_size=len(valid_set), shuffle=True, num_workers=4,
                              pin_memory=cuda)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=cuda)

model_vae = VAE(2).to(device)
optimizer = optim.Adam(model_vae.parameters(), lr=lr)
loss_function = elbo_loss


#%%


from tqdm import tqdm

valid_loss = float("inf")
elbo = []
recon = []
regul = []

with tqdm(total=epochs * nTrain, unit='images', desc='Training Model') as pbar:
    for epoch in range(1, epochs + 1):
        train_loss = 0

        model_vae.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            xHat, mu, logvar = model_vae(data)

            loss = loss_function(xHat, data, mu, logvar, KLD_weight)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pbar.update(len(data))
            pbar.set_postfix_str(f'train loss: {loss.item():>3f}, val loss: {valid_loss:>3f}, epoch: {epoch}')

        valid_loss = 0
        recon_loss = 0
        regul_loss = 0
        model_vae.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(valid_loader):
                data = data.to(device)
                xHat, mu, logvar = model_vae(data)
                valid_loss += loss_function(xHat, data, mu, logvar, KLD_weight).item()
                recon_loss += F.binary_cross_entropy(xHat, data, reduction='mean').item()
                regul_loss += KLD_weight * torch.mean(- 0.5 * torch.sum(1 + logvar - mu ** 2 - torch.exp(logvar), -1)).item()

                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data.view(len(data), 1, 28, 28)[:n],
                                          xHat.view(len(data), 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                             'results/reconstruction_vae_' + str(epoch) + '.png', nrow=n)

        valid_loss /= len(valid_loader)

        elbo.append(valid_loss)
        recon.append(recon_loss)
        regul.append(regul_loss)


fig, ax = plt.subplots()

ln1 = ax.plot(list(range(epochs)), elbo, marker='o', label="elbo")
ln2 = ax.plot(list(range(epochs)), recon, marker='o', label="reconstruction loss")
ax2 = ax.twinx()
ln3 = ax2.plot(list(range(epochs)), regul, marker='o', label="regularization", c='green')

lns = ln1 + ln2 + ln3

ax.legend(lns, [l.get_label() for l in lns])
ax.set_xlabel("epoch")
ax.set_ylabel("validation elbo and reconstruction loss")
ax2.set_ylabel("validation regularization loss")
plt.show()


#%%

# Use converged model to evaluate on train and test set.

model_vae.eval()

with torch.no_grad():
    xHat_train, mu_train, logvar_train = model_vae(X_train_reduced.to(device))
    xHat_train = xHat_train.cpu()
    mu_train = mu_train.cpu()
    logvar_train = logvar_train.cpu()
    print(f"Train elbo loss: {loss_function(xHat_train, X_train_reduced, mu_train, logvar_train, KLD_weight).item():.4f}")
    print(f"Train recon loss: {F.binary_cross_entropy(xHat_train, X_train_reduced, reduction='mean').item():.4f}")
    print(f"Train regul loss: {KLD_weight * torch.mean(- 0.5 * torch.sum(1 + logvar_train - mu_train ** 2 - torch.exp(logvar_train), -1)).item():.4f}")
    print()
    
    xHat_test, mu_test, logvar_test = model_vae(X_test_reduced.to(device))
    xHat_test = xHat_test.cpu()
    mu_test = mu_test.cpu()
    logvar_test = logvar_test.cpu()
    print(f"Test elbo loss: {loss_function(xHat_test, X_test_reduced, mu_test, logvar_test, KLD_weight).item():.4f}")
    print(f"Test recon loss: {F.binary_cross_entropy(xHat_test, X_test_reduced, reduction='mean').item():.4f}")
    print(f"Test regul loss: {KLD_weight * torch.mean(- 0.5 * torch.sum(1 + logvar_test - mu_test ** 2 - torch.exp(logvar_test), -1)).item():.4f}")
    print()

clustering(mu_train, y_train_reduced, mu_test, y_test_reduced)

#%%

# 6. Generative Sampling:

z_new = torch.randn((32, 2))

model_vae.eval()
with torch.no_grad():
    X_new = model_vae.decode(z_new.to(device)).cpu()


fig, axs = plt.subplots(4, 8)
i = 0

for row in axs:
    for ax in row:
        ax.imshow(X_new[i].view(28, 28), cmap="Greys")
        ax.axis('off')
        i += 1

fig.tight_layout()
plt.show()
