from data_processing import get_dataloader

from models import TwoLayerNN, train, test
import numpy as np
import random
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split



def test_linear_nn():
    
    test_frac = 0.0
    BATCH_SIZE = 1  # batch size
    NUM_EPOCH = 10  # number of training epochs
    LR = 0.0003  # learning rate
    HIDDEN_SIZE = 1
    
    data_path_X = "/users/bsun14/data/bsun14/BRATS_TCGA_GBM_all_niftis"
    data_path_Y = "/users/bsun14/data/bsun14/TCIA_manual_segmentations"
    
    train_loader, test_loader = get_dataloader(data_path_X, data_path_Y)

    model = TwoLayerNN(hidden_size = HIDDEN_SIZE, input_size = 155*240*240)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_func = nn.MSELoss()
    losses = train(model, train_loader, loss_func, optimizer, NUM_EPOCH)
    
    test_loss = test(model, test_loader, loss_func)
    print(test_loss)
    
test_linear_nn()

    






