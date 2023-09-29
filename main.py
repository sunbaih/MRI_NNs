from data_processing import get_dataloader

from models import TwoLayerNN, train, test, volume_CNN
import numpy as np
import random
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split

def set_device(): 
    torch.cuda.is_available()
    device = torch.device("cuda:0")
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    return device 

def test_linear_nn(device):
    
    test_frac = 0.2
    BATCH_SIZE = 30  # batch size
    NUM_EPOCH = 10  # number of training epochs
    LR = 0.01  # learning rate
    HIDDEN_SIZE = 1
    
    data_path_X = "/users/bsun14/data/bsun14/BRATS_TCGA_GBM_all_niftis"
    data_path_Y = "/users/bsun14/data/bsun14/TCIA_manual_segmentations"
    
    train_loader, test_loader = get_dataloader(data_path_X, data_path_Y, batch_size = BATCH_SIZE, test_size= test_frac)

    #model = TwoLayerNN(hidden_size = HIDDEN_SIZE, input_size = 3*10*10).to(device)
    model = volume_CNN(in_channels = 1, sample_size = BATCH_SIZE, sample_duration = 24*24*24 )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_func = nn.MSELoss()
    losses = train(device, model, train_loader, loss_func, optimizer, NUM_EPOCH)
    
    test_loss = test(device, model, test_loader, loss_func)
    print(test_loss)

device = set_device()
test_linear_nn(device)

    






