
import torch
from torch import nn
from data_processing import reconstruct_nifti, average_of_three_entries
from tqdm import tqdm


import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True):
        """
        Initializes the early stopping callback.

        Parameters:
            patience (int): The number of epochs with no improvement after which training will be stopped.
            delta (float): The minimum change in the monitored metric to qualify as an improvement.
            verbose (bool): If True, prints a message when early stopping is triggered.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, current_metric):
        """
        Checks if training should be stopped based on the current metric value.

        Parameters:
            current_metric (float): The current value of the monitored metric.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.best_metric is None:
            self.best_metric = current_metric
        elif current_metric < self.best_metric - self.delta:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                self.early_stop = True
        return self.early_stop



class TwoLayerNN(nn.Module):

    def __init__(self, hidden_size, input_size=3*10*10):
        """
        Initializes model layers.
        :param input_size: number of input feautures (also equals output features in this case)
        """
        super().__init__()
        
        self.hidden_size = hidden_size 
        
        in_size = int(input_size/3)
        
        self.dropout_1 = torch.nn.Dropout(0.5)
        self.layer_1 = torch.nn.Linear(in_size, self.hidden_size)
        self.dropout_2 = torch.nn.Dropout(0.5)
        self.activation = torch.nn.LeakyReLU(0.01)
        self.layer_2 = torch.nn.Linear(self.hidden_size, input_size)

    def forward(self, X):
        
        X = average_of_three_entries(X)
        X = self.dropout_1(X)
        z1 = self.layer_1(X)
        z1 = self.dropout_2(z1)
        a1 = self.activation(z1)
        z2 = self.layer_2(a1)
        return z2


class volume_CNN_8(nn.Module):
    
    def __init__(self, in_channels, sample_size, sample_duration):
        
        super().__init__()
        
        self.group1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))
        
        """
        last_duration = int(math.floor(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        
        #fc_in_size = 512 * last_duration * last_size * last_size
        fc_in_size = 1024
        
        self.fc1 = nn.Sequential(
            nn.Linear(fc_in_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(4096, sample_duration))  
        """
        
        # Group 5
        self.group5_upsample = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU())
        
        # Group 4
        self.group4_upsample = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU())
        
        # Group 3
        self.group3_upsample = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU())
        
        # Group 2
        self.group2_upsample = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU())
        """
        # Group 1
        self.group1_upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 1, kernel_size=3, padding=1))
            """
        # last layer kernel, stride, padding chosen to get output image size = input image size = [155, 240, 240]
        self.group1_upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 1, kernel_size=(3, 2, 1), stride = (1, 2, 2), padding=(1, 8, 2))) 

    def forward(self, x):
        
        # need to add dimension = 1 to incorporate number of channels 
        
        
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)

        print(out.shape)
        """
        out = out.view(-1)
        print(out.shape)
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        """
        out = self.group5_upsample(out)
        print("after 5", out.shape)
        out = self.group4_upsample(out)
        print("after 4", out.shape)
        out = self.group3_upsample(out)
        print("after 3", out.shape)
        out = self.group2_upsample(out)
        print("after 2", out.shape)
        out = self.group1_upsample(out)
        print("after 1", out.shape)

        return out
    
class volume_CNN(nn.Module):
    
    def __init__(self, in_channels, sample_size, sample_duration):
        
        super().__init__()
        
        self.group1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size= 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        
        """

        last_duration = int(math.floor(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        
        #fc_in_size = 512 * last_duration * last_size * last_size
        fc_in_size = 1024
        
        self.fc1 = nn.Sequential(
            nn.Linear(fc_in_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(4096, sample_duration))  
        
        
        """
        
        # Group 4
        self.group4_upsample = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(4,4,5), stride=2, padding =1),
            nn.BatchNorm3d(256),
            nn.ReLU())
        
        # Group 3
        self.group3_upsample = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(4,4,5), stride=2, padding =1),
            nn.BatchNorm3d(128),
            nn.ReLU())
        
        # Group 2
        self.group2_upsample = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=(4,4,5), stride =2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU())
        
        # Group 1
        self.group1_upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 1, kernel_size=(4,4,3), stride = 2,padding=(1, 1, 2)))
    

    def forward(self, x):
        
        # need to add dimension = 1 to incorporate number of channels 
        
        
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
              
        print(out.shape)
        """
        out = out.view(-1)
        print(out.shape)
        
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        """
        out = self.group4_upsample(out)
        print("4", out.shape)
        out = self.group3_upsample(out)
        print("3", out.shape)
        out = self.group2_upsample(out)
        print("2", out.shape)
        out = self.group1_upsample(out)
        print("final", out.shape)

        return out

def train(device, model, dataloader, loss_func, optimizer, num_epoch,  print_info=True):
    epoch_average_losses = []
    model.train()
    early_stopping = EarlyStopping(patience=5, delta=0.001, verbose=True)
    best_training_loss = np.inf
    
    for epoch in range(num_epoch):

        epoch_loss_sum = 0

        for batch in enumerate(tqdm(dataloader)):
            patient_id, X, Y = batch[1]
            
            if isinstance(model, TwoLayerNN):
                X = X.reshape(-1)
                Y = Y.reshape(-1)
                
            if isinstance(model, volume_CNN):
                X = X.unsqueeze(1)
                Y = Y.unsqueeze(1)
                
            X = X.float().to(device)
            Y = Y.float().to(device)

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
        
        training_loss = epoch_loss_sum/len(dataloader.dataset)
        epoch_average_losses.append(training_loss)
        
        # Check if training loss is the best so far
        if training_loss < best_training_loss:
            best_training_loss = training_loss

        if early_stopping(training_loss):
            break  # Stop training

        # Print the loss after every epoch. Print accuracies if specified
        if print_info:
            print('Epoch: {} | Loss: {:.4f} '.format(epoch, epoch_loss_sum / len(dataloader.dataset)))
            
    return epoch_average_losses


        
def test(device, model, dataloader, loss_func):

    loss_sum = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            patient_ids, X, Y = batch
            
            if isinstance(model, TwoLayerNN):
                X = X.reshape(-1)
                Y = Y.reshape(-1)
                
            if isinstance(model, volume_CNN):
                X = X.unsqueeze(1)
                Y = Y.unsqueeze(1)
            
            X = X.float().to(device)
            Y = Y.float().to(device)
                
            output = model.forward(X)
            loss = loss_func(output, Y)
            loss_sum += loss.item()*X.shape[0]
            
            if len(patient_ids)==1:
                reconstruct_nifti(patient_ids[0], Y)
            else: 
                for i in range(len(patient_ids)):
                    reconstruct_nifti(patient_ids[i], Y[i])
                    
                
    return loss_sum/len(dataloader.dataset)