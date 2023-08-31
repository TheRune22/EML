import os
import random
from typing import Sized

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image

from sklearn.model_selection import train_test_split

# Fix random seed
random.seed(2)
np.random.seed(2)
torch.manual_seed(2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"Using device {device}")


input_width = 565
input_height = 584


# Class and function definitions
class ImageDataset(Dataset, Sized):
    def __init__(self, X_paths, y_paths):
        self.X_paths = X_paths
        self.y_paths = y_paths
        self.X_images = [transforms.ToTensor()(Image.open(X_path)) for X_path in self.X_paths]
        self.y_images = [transforms.ToTensor()(Image.open(y_path)) for y_path in self.y_paths]

        X_numpy = np.array([x.numpy() for x in self.X_images])
        self.X_mean = np.mean(X_numpy, (0, -1, -2))
        self.X_std = np.std(X_numpy, (0, -1, -2))

        self.X_mean = torch.tensor(self.X_mean)
        self.X_std = torch.tensor(self.X_std)

    def __len__(self):
        return len(self.X_images)

    def __getitem__(self, idx):
        return self.X_images[idx], self.y_images[idx]


class RandomlyAugmentedDataset(Dataset):
    def __init__(self, base_dataset: (Dataset, Sized), size=None, flip=True,
                 rotate=True, brightness=True, contrast=True, saturation=True):
        self.base_dataset = base_dataset
        self.base_len = len(self.base_dataset)

        if size is None:
            size = self.base_len
        self.size = size

        self.flip = flip
        self.rotate = rotate
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        X_img, y_img = self.base_dataset[idx % self.base_len]

        if self.flip:
            if random.random() < 1:
                X_img = TF.hflip(X_img)
                y_img = TF.hflip(y_img)

        if self.rotate:
            angle = random.uniform(-180, 180)
            X_img = TF.rotate(X_img, angle)
            y_img = TF.rotate(y_img, angle)

        if self.brightness:
            brightness_factor = random.normalvariate(1, 0.2 / 4)
            X_img = TF.adjust_brightness(X_img, brightness_factor)

        if self.contrast:
            contrast_factor = random.normalvariate(1, 0.2 / 4)
            X_img = TF.adjust_contrast(X_img, contrast_factor)

        if self.saturation:
            saturation_factor = random.normalvariate(1, 0.2 / 4)
            X_img = TF.adjust_saturation(X_img, saturation_factor)

        return X_img, y_img


def get_paths(dir):
    files = os.listdir(dir)
    files.sort()
    return [os.path.join(dir, file) for file in files]


def train(model, optimizer, loss_fn, dataloader_train, dataloader_val, batch_size,
          epoch_size, epochs):
    losses_train = []
    losses_val = []
    loss_val = float('inf')
    model.train()

    # Progress bar
    with tqdm(total=epoch_size * epochs, unit='images', desc='Training Model') as pbar:
        for i, (X, y) in enumerate(dataloader_train):
            X_len = len(X)
            X = X[:, [1]].to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            if device == 'cuda':
                del X
                torch.cuda.empty_cache()

            # Backpropagation
            y = y.to(device)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            if device == 'cuda':
                del y
                torch.cuda.empty_cache()

            loss_item = loss.item()
            pbar.update(X_len)
            pbar.set_postfix_str(f'train loss: {loss_item:>3f}, val loss: {loss_val:>3f}')
            losses_train.append(loss_item)

            if (i * batch_size) % epoch_size == 0:
                model.eval()
                for i, (X, y) in enumerate(dataloader_val):
                    with torch.no_grad():
                        X = X[:, [1]].to(device)
                        y_pred = model(X)
                        if device == 'cuda':
                            del X
                            torch.cuda.empty_cache()

                        y = y.to(device)
                        loss = loss_fn(y_pred, y)
                        if device == 'cuda':
                            del y
                            torch.cuda.empty_cache()

                        loss_item = loss.item()
                        loss_val = loss_item
                        losses_val.append(loss_item)
                model.train()

    fig, ax = plt.subplots()
    ax.plot(range(batch_size, (len(losses_train) + 1) * batch_size, batch_size), losses_train,
            marker='.', label="Training loss")
    ax.plot(range(epoch_size, (len(losses_val) + 1) * epoch_size, epoch_size), losses_val,
            marker='.', label="Validation loss")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Dice Loss")
    ax.legend()
    plt.show()


def dice_sim(A, B):
    # Reduces using mean if multiple 2d tensors are given, for example caused by batching
    return (2 * torch.sum(A * B, (-1, -2)) /
            (torch.sum(B, (-1, -2)) + torch.sum(A, (-1, -2)))).mean()


def dice_loss(A, B):
    # Reduces using mean if multiple 2d tensors are given, for example caused by batching
    return 1 - ((2 * torch.sum(A * B, (-1, -2)) + 1) /
                ((torch.sum(B ** 2, (-1, -2)) + torch.sum(A ** 2, (-1, -2))) + 1)).mean()


def eval_model(model, data):
    total_dice_loss = 0
    total_BCE_loss = 0
    total_dice_sim = 0

    bce = nn.BCELoss(reduction='mean')
    for X, y in DataLoader(data, batch_size=1):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(np.moveaxis(X[0].numpy(), 0, -1))
        ax2.imshow(np.moveaxis(y[0].numpy(), 0, -1))

        with torch.no_grad():
            model.eval()
            y_pred = model(X[:, [1]].to(device))
            y = y.to(device)

            loss_BCE = bce(y_pred, y)
            total_BCE_loss += loss_BCE
            print(f"BCE Loss: {loss_BCE}")

            loss_dice = dice_loss(y_pred, y)
            total_dice_loss += loss_dice
            print(f"Dice Loss: {loss_dice}")

            dice = dice_sim(y, torch.round(y_pred))
            total_dice_sim += dice
            print(f"Dice similarity: {dice}")

        y_pred = y_pred.cpu()
        ax3.imshow(np.moveaxis(y_pred[0].numpy(), 0, -1))
        ax4.imshow(np.moveaxis(torch.round(y_pred[0]).numpy(), 0, -1))

        plt.show()

    print(f"Mean BCE Loss: {total_BCE_loss / len(data)}")
    print(f"Mean Dice Loss: {total_dice_loss / len(data)}")
    print(f"Mean Dice similarity: {total_dice_sim / len(data)}")


class SkipConnection(nn.Module):
    def __init__(self, *args: nn.Module):
        super().__init__()
        if len(args) == 1:
            self.mid = args[0]
        else:
            self.mid = nn.Sequential(*args)

    def forward(self, x):
        return torch.cat([self.mid(x), x], 1)


def block(activation, **conv_args):
    # TODO: remove sequential?
    return nn.Sequential(
        nn.Conv2d(**conv_args),
        nn.BatchNorm2d(conv_args['out_channels']),
        activation(),
    )


def inner_unet_modules(levels, activation, input_height, input_width, in_channels, out_channels):
    def double_block(in_channels, out_channels):
        # TODO: allow setting arbitrary parameters, or just kernel_size?
        return [
            block(activation, in_channels=in_channels, out_channels=out_channels,
                  kernel_size=3, padding='same'),
            block(activation, in_channels=out_channels, out_channels=out_channels,
                  kernel_size=3, padding='same'),
        ]

    # Stopping condition
    if levels == 0:
        # Innermost level, double block
        return double_block(in_channels, out_channels)
    else:
        return [
            *double_block(in_channels, out_channels),
            SkipConnection(
                nn.MaxPool2d(2),
                # TODO: move int to usage?
                *inner_unet_modules(levels - 1, activation, int(input_height / 2), int(input_width / 2), out_channels, out_channels * 2),
                # TODO: should this actually halve channels?
                # nn.ConvTranspose2d(in_channels=out_channels * 2, out_channels=out_channels,
                nn.ConvTranspose2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                                   kernel_size=2, stride=2,
                                   output_padding=(input_height % 2,input_width % 2)),
            ),
            # TODO: out_channels*2 instead of out_channels*3?
            # *double_block(out_channels * 2, out_channels),
            *double_block(out_channels * 3, out_channels),
        ]


def show_channels(img_dir):
    files = os.listdir(img_dir)

    files.sort(key=lambda x: int(x[:x.find('_')]))

    # fig, axs = plt.subplots(len(files), 4, figsize=(4, 15), dpi=80)

    for i, file in enumerate(files):
        if i % 5 == 0:
            fig, axs = plt.subplots(5, 4, figsize=(4, 5), dpi=500)

        img = Image.open(os.path.join(img_dir, file))
        X = transforms.ToTensor()(img)

        axs[i % 5, 0].imshow(np.moveaxis(X.numpy(), 0, -1))
        red = X.numpy().copy()
        red[[1, 2]] = 0
        axs[i % 5, 1].imshow(np.moveaxis(red, 0, -1))

        green = X.numpy().copy()
        green[[0, 2]] = 0
        axs[i % 5, 2].imshow(np.moveaxis(green, 0, -1))

        blue = X.numpy().copy()
        blue[[0, 1]] = 0
        axs[i % 5, 3].imshow(np.moveaxis(blue, 0, -1))

        axs[i % 5, 0].set_title(file)

        for ax in axs[i % 5]:
            ax.axis('off')

        if i % 5 == 4:
            fig.tight_layout()
            plt.show()


# Loading and splitting data
X_train_dir = "Images_train"
y_train_dir = "Labels_train"

X_paths = get_paths(X_train_dir)
y_paths = get_paths(y_train_dir)

X_paths_train_val, X_paths_test, y_paths_train_val, y_paths_test = train_test_split(
    X_paths, y_paths, test_size=0.1, random_state=2)
X_paths_train, X_paths_val, y_paths_train, y_paths_val = train_test_split(
    X_paths_train_val, y_paths_train_val, test_size=0.1, random_state=2)

base_data_train = ImageDataset(X_paths_train, y_paths_train)
base_data_val = ImageDataset(X_paths_val, y_paths_val)
base_data_test = ImageDataset(X_paths_test, y_paths_test)


if device == 'cuda':
    torch.cuda.empty_cache()

# Definition of model
model = nn.Sequential(
    transforms.Normalize(base_data_train.X_mean[1], base_data_train.X_std[1]),

    *inner_unet_modules(5, nn.ReLU, input_height, input_width, 1, 32),

    nn.Conv2d(in_channels=32, out_channels=1,
              kernel_size=1, padding='same'),
    nn.Sigmoid(),
).to(device)

#%% Additional hyperparameters

batch_size = 2
epoch_size = batch_size * len(base_data_train)
epochs = 100
learning_rate = 5 * 1e-4 * batch_size

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# loss_fn = nn.BCELoss(reduction='mean')
loss_fn = dice_loss

data_train = RandomlyAugmentedDataset(base_data_train, size=epoch_size * epochs,
                                      flip=True, rotate=True, brightness=True,
                                      contrast=True, saturation=True)
dataloader_train = DataLoader(data_train, batch_size=batch_size, num_workers=4,
                              pin_memory=(device == 'cuda'), shuffle=True)

data_val = RandomlyAugmentedDataset(base_data_val, size=10,
                                    flip=True, rotate=True, brightness=False,
                                    contrast=False, saturation=False)
dataloader_val = DataLoader(data_val, batch_size=len(data_val), num_workers=4,
                            pin_memory=(device == 'cuda'))

#%% Training model

train(model, optimizer, loss_fn, dataloader_train, dataloader_val,
      batch_size, epoch_size, epochs)

#%%

print("\nEvaluating on training data")
eval_model(model, base_data_train)

#%%

print("\nEvaluating on validation data")
eval_model(model, base_data_val)

#%%

print("\nEvaluating on test data")
eval_model(model, base_data_test)

#%%

X_test_dir = "Images_test"

for file in get_paths(X_test_dir):
    img = Image.open(file)
    X = transforms.ToTensor()(img).unsqueeze(0)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(np.moveaxis(X[0].numpy(), 0, -1))
    with torch.no_grad():
        model.eval()
        y_pred = model(X[:, [1]].to(device))
    y_pred = y_pred.cpu()
    ax2.imshow(np.moveaxis(torch.round(y_pred[0]).numpy(), 0, -1))
    ax3.imshow(np.moveaxis(y_pred[0].numpy(), 0, -1))
    plt.show()

    torchvision.utils.save_image(torch.round(y_pred[0]), file.split('/')[1])


#%%

show_channels(X_train_dir)

#%%

show_channels(X_test_dir)
