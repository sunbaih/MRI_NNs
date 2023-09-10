
import torch
from torch import nn

class TwoLayerNN(nn.Module):

    def __init__(self, hidden_size, input_size=240*240*150):
        """
        Initializes model layers.
        :param input_size: number of input feautures (also equals output features in this case)
        """
        super().__init__()

        self.hidden_size = hidden_size 
        self.layer_1 = torch.nn.Linear(input_size, self.hidden_size)
        print(self.layer_1.weight.dtype)
        self.activation = torch.nn.LeakyReLU(0.01)
        self.layer_2 = torch.nn.Linear(self.hidden_size, input_size)

    def forward(self, X):

        z1 = self.layer_1(X)
        a1 = self.activation(z1)
        z2 = self.layer_2(a1)
        return z2
    
def train(model, dataloader, loss_func, optimizer, num_epoch,  print_info=True):
    print(train)
    epoch_average_losses = []
    model.train()

    for epoch in range(num_epoch):

        epoch_loss_sum = 0

        for batch in dataloader:
            patient_id, X, Y = batch
            X = X.reshape(-1).float()
            Y = Y.reshape(-1).float()
            output = model.forward(X)
            optimizer.zero_grad()
            loss = loss_func(output, Y)
            loss.backward()
            optimizer.step()

            #Increase epoch_loss_sum by (loss * #samples in the current batch)
            #  Use loss.item() to get the python scalar of loss value because the output of
            #   loss function also contains gradient information, which takes a lot of memory.
            #  Use X.shape[0] to get the number of samples in the current batch.
            epoch_loss_sum += loss.item()*X.shape[0]

        epoch_average_losses.append(epoch_loss_sum/len(dataloader.dataset))

        # Print the loss after every epoch. Print accuracies if specified
        if print_info:
            print('Epoch: {} | Loss: {:.4f} '.format(epoch, epoch_loss_sum / len(dataloader.dataset)))
            
    return epoch_average_losses


def test(model, dataloader, loss_func):

    loss_sum = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            count +=1 
            output = model.forward(X)
            loss = loss_func(output, Y)
            loss_sum += loss.item()*X.shape[0]             

    return loss_sum/len(dataloader.dataset)


