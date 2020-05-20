#Import the Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

#Setup Plotting Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Only if executing within Jupyter Notebook
# %matplotlib inline



#Enables Gradient Tracking
torch.set_grad_enabled(True)



#Create the neural architecture class
class Network(nn.Module):

    def __init__(self):

        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 12 * 16, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 50)
        self.out = nn.Linear(in_features = 50, out_features = 10)

    def forward(self, t):

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)

        return t

def get_num_correct(preds, labels):

    return int(preds.argmax(dim = 1).eq(labels).sum().item())


def get_all_preds(model, loader):
    all_preds = torch.tensor([])

    for batch in loader:

        images, labels = batch

        preds = model(images)

        all_preds = torch.cat((all_preds, preds), dim = 0)

    return all_preds


def main():

    #Hyperparameters
    batch_size = 100
    lr = 0.01

    ann = Network()
    print("[+] Neural Network Instantiated")

    #Create Training Dataset
    train_set = torchvision.datasets.FashionMNIST(

        root = "./data/FashionMNIST",
        train = True,
        download = True,
        transform = transforms.Compose([transforms.ToTensor()])
    )

    print("[+] Dataset Created")
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    optimizer = optim.Adam(ann.parameters(), lr = lr)

    print("[+] DataLoader Created")


    print("\n[+] Instantiating Training Loop")
    epochs = 1

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch

            preds = ann(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print("")
        print("epoch: ", epoch, "|"," total_correct: ", total_correct, "|"," loss: ", total_loss, "|", "accuracy: ", total_correct / len(train_set))

    print("\n[+]Training Loop Complete")

    #Compute The Predictions without generating Computation Graph
    with torch.no_grad():
        prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=1000)
        train_preds = get_all_preds(ann, prediction_loader)

    #Generate a tensor to contain predictions and orignal targets
    stacked = torch.stack((train_set.train_labels, train_preds.argmax(dim = 1)), dim = 1)

    #Create an empty confusion matrix
    cmt = torch.zeros(10, 10, dtype=torch.int64)

    #Populate the empty confusion matrix
    for p in stacked:
        j, k = p.tolist()

        cmt[j, k] += 1

    #Populate Labels
    y_actual = y_pred = ["Ankle Boot", "Bag", "Sneaker", "Shirt", "Sandal", "Coat", "Dress", "Pullover", "Trouser", "T-Shirt/Top"]

    #Create a confusion dataframe
    df = pd.DataFrame(cmt, index = y_actual, columns = y_pred, dtype=int)

    print("[+] Confusion Matrix DataFrame Created")
    print(df)

    df.index.name = 'Actual'
    df.columns.name = 'Predicted'
    plt.figure(figsize=(30, 20))
    sns.set(font_scale=0.5)
    sns.heatmap(df, cmap = 'Blues', annot=True,annot_kws={"size": 8}, fmt = 'd')
    plt.show()

if __name__ == "__main__":

    main()

