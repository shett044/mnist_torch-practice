from typing import Any
import idx2numpy
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# bash> tensorboard --logdir=runs (Port forward 6006)

EXP_LOC = "runs/mnist"
writer = SummaryWriter(EXP_LOC)

# Custom packge
import utils.torch_helper as uth
import model as mdl

# import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# MNSIT
class XParam(uth.Param):
    pass


# Custom Ops
class XOPS(uth.OPS):
    uth.OPS.NUM_EPOCHS = 3
    uth.OPS.LR = 0.001
    pass

# Cannot download datasets
# train_dataset = torchvision.datasets.MNIST(root = './data', train =True, transform= transforms.ToTensor(), download= False)
# test_dataset = torchvision.datasets.MNIST(root = './data', train =False, transform= transforms.ToTensor(), download= False)

class MNIST(Dataset):
    def __init__(self, train_file: str, label_file: str, transform: callable = None):
        self.X = idx2numpy.convert_from_file(train_file).astype(np.float32)
        # Add channel
        self.X = np.expand_dims(self.X, axis=1)
        self.y = idx2numpy.convert_from_file(label_file).astype(np.uint8)

        if transform:
            self.X = transform(self.X).to(device)
            self.y = transform(self.y).to(device)

    def __getitem__(self, index) -> Any:
        return self.X[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]

# Unzip the data from manual download
raw_path = "data/MNIST/raw"

train_dataset = MNIST(
    os.path.join(raw_path, "train-images-idx3-ubyte"),
    os.path.join(raw_path, "train-labels-idx1-ubyte"),
    uth.ToTensor(),
)
test_dataset = MNIST(
    os.path.join(raw_path, "t10k-images-idx3-ubyte"),
    os.path.join(raw_path, "t10k-labels-idx1-ubyte"),
    uth.ToTensor(),
)

train_loader = DataLoader(
    dataset=train_dataset, batch_size=XOPS.BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=XOPS.BATCH_SIZE)

ex = iter(train_loader)
samples, lbl = ex.next()
samples = samples.cpu()
print(samples.shape, lbl.shape)
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap="gray")
import torchvision

img_grid = torchvision.utils.make_grid(samples)
writer.add_image("mnist_images", img_grid)
writer.close()

# sys.exit()

model = mdl.NeuralNet(XParam.INPUT_SIZE, XParam.OUTPUT_SIZE, XParam.HIDDEN_SIZE)

# Pipeline
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adagrad(model.parameters(), lr=XOPS.LR)
writer.add_graph(model.cpu(), samples.reshape(-1, XParam.INPUT_SIZE))
writer.close()
model.to(device)
# sys.exit()
n_steps = len(train_loader)
running_loss = 0.0
running_corr = 0

print(f"{XOPS.NUM_EPOCHS=}")
for e in range(XOPS.NUM_EPOCHS):
    for i, (samples, label) in enumerate(train_loader):
        samples = samples.reshape(-1, XParam.INPUT_SIZE)
        label = label
        # Forward
        output = model(samples)
        loss = criterion(output, label)

        # backward
        optim.zero_grad()
        loss.backward()
        optim.step()
        with torch.no_grad():
            running_loss += loss.item()
            _, preds = torch.max(output, 1)
            running_corr += (preds == label).sum().item()

            if (i + 1) % 100 == 0:
                global_i = e * n_steps + i + 1
                print(
                    f"epoch {e + 1}/{XOPS.NUM_EPOCHS}, step {i + 1}/ {n_steps} loss = {loss.item():.3f}"
                )
                writer.add_scalar("training loss", running_loss / 100, global_i)
                writer.add_scalar("training acc", running_corr / 100, global_i)
                running_loss = 0.0
                running_corr = 0


# test
pr_labels = []
pr_preds = []
with torch.no_grad():
    n_correct, n_samples = 0, 0
    for images, labels in test_loader:
        images = images.reshape(-1, XParam.INPUT_SIZE)
        labels = labels
        output = model(images)

        _, preds = torch.max(output, 1)
        n_samples = labels.shape[0]
        n_correct = (preds == labels).sum().item()
        pr_labels.append(labels.cpu())
        pr_preds.append(F.softmax(output.cpu(), dim=1))

    pr_preds = torch.cat(pr_preds)
    pr_labels = torch.cat(pr_labels)
    # Add PR curve
    for class_ in range(XParam.OUTPUT_SIZE):
        writer.add_pr_curve(
            f"PR Curve : {class_=}", pr_labels == class_, pr_preds[:, class_]
        )
        writer.close()

    print(f"Accuracy : {n_correct/n_samples}")
model_save_path = os.path.join(EXP_LOC, "final_model.pth")
print(f"Model saved in {model_save_path}")
torch.save(model.state_dict(), model_save_path)