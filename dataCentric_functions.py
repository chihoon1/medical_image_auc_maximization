from sklearn import metrics
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG, Adam, SGD
from libauc.metrics import auc_roc_score

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
#from pytorchtools import EarlyStopping
import sys


# Gaussian Noise Data Augmentation function
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



# pytorch based model training function
def train(model, loss_fn, optimizer, total_epochs, trainloader, trainloader_eval, testloader,
          patience=15, decay_epochs=None):
    # param: model=pytorch NN model
    # param: loss_fn = loss function
    # param: optimizer = pytorch optimizer
    # param: total_epochs(int) = total epochs for training
    # param: trainloader(pytorch data loader). train dataset loader used for model training
    # param: trainloader_eval(pytorch data loader). train dataset loader used for evaluation
    # param: testloader(pytorch data loader)
    # param: patience - how long to wait after last time validation loss improved
    # param: decay_epochs is decay epochs range for decreasing the learning rate (default=None, then not used)
    # return a tuple of train_log and test_log. log arrays contain auc values from each epoch

    print('Start Training')
    print('-' * 30)

    train_log = []
    test_log = []

    # initialize the early_stopping object
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    no_val_change_count = 0
    last_val_auc = 0
    for epoch in range(total_epochs):
        if decay_epochs is not None and epoch in decay_epochs:
            optimizer.update_regularizer(decay_factor=10)  # decrease learning rate by 10x & update regularizer

        train_loss = []
        model.train()
        for data, targets in trainloader:
            #print(len(data))
            # data, targets  = data.cuda(), targets.cuda()
            #print(data.shape, targets.shape)
            y_pred = model(data)
            y_pred = torch.sigmoid(y_pred)
            loss = loss_fn(y_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # evaluation on train & test sets
        model.eval()
        train_pred_list = []
        train_true_list = []
        for train_data, train_targets in trainloader_eval:
            train_pred = model(train_data)
            train_pred = torch.sigmoid(train_pred)
            # train_pred_list.append(train_pred.cpu().detach().numpy())
            # train_true_list.append(train_targets.numpy())
            train_pred_list.append(train_pred.detach().numpy())
            train_true_list.append(train_targets.numpy())
        train_true = np.concatenate(train_true_list)
        train_pred = np.concatenate(train_pred_list)
        train_auc = np.mean(auc_roc_score(train_true, train_pred))
        train_loss = np.mean(train_loss)

        test_pred_list = []
        test_true_list = []
        for test_data, test_targets in testloader:
            test_pred = model(test_data)
            test_pred = torch.sigmoid(test_pred)
            test_pred_list.append(test_pred.detach().numpy())
            test_true_list.append(test_targets.numpy())
        test_true = np.concatenate(test_true_list)
        test_pred = np.concatenate(test_pred_list)
        val_auc = np.mean(auc_roc_score(test_true, test_pred))
        model.train()

        # print results
        print("epoch: %s, train_loss: %.4f, train_auc: %.4f, val_auc: %.4f, lr: %.4f" % (
        epoch, train_loss, train_auc, val_auc, optimizer.lr))
        train_log.append(train_auc)
        test_log.append(val_auc)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        # early_stopping(valid_auc, model)
        if val_auc <= last_val_auc  or abs(val_auc - last_val_auc) <= 1e-3*5:
            # validation auc decreasing or the difference of validation auc between epochs indicate the stopping point
            no_val_change_count += 1
        else:
            no_val_change_count = 0

        print(no_val_change_count)
        if no_val_change_count >= patience:
            print("Early stopping")
            break

        last_val_auc = val_auc
        # if early_stopping.early_stop:
        #    print("Early stopping")
        #    break
    return train_log, test_log


def multilabel_train(model, loss_fn, optimizer, total_epochs, trainloader, trainloader_eval, valloader,
          patience=12, decay_epochs=None):
    # this training used for multilabel ResNet training
    # param: model=pytorch NN model
    # param: loss_fn = loss function
    # param: optimizer = pytorch optimizer
    # param: total_epochs(int) = total epochs for training
    # param: trainloader(pytorch data loader). train dataset loader used for model training
    # param: trainloader_eval(pytorch data loader). train dataset loader used for evaluation
    # param: valloader(pytorch data loader)
    # param: patience - how long to wait after last time validation loss improved
    # param: decay_epochs is decay epochs range for decreasing the learning rate (default=None, then not used)

    print('Start Training')
    print('-' * 30)

    train_log = []
    val_log = []

    # initialize the early_stopping object
    no_val_change_count = 0
    last_val_auc = 0
    #best_val_auc = 0
    for epoch in range(total_epochs):
        #if epoch > 0:
        #    optimizer.update_regularizer(decay_factor=10)
        if decay_epochs is not None and epoch in decay_epochs:
            optimizer.update_regularizer(decay_factor=10)  # decrease learning rate by 10x & update regularizer

        train_loss = []
        model.train()
        for idx, data in enumerate(trainloader):
            train_data, train_labels = data
            #print(train_data.shape, train_labels.shape)
            #train_data, train_labels = train_data.cuda(), train_labels.cuda()
            y_pred = model(train_data)
            y_pred = torch.sigmoid(y_pred)
            loss = loss_fn(y_pred, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # validation
            if idx % 400 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = []
                    test_true = []
                    for jdx, data in enumerate(valloader):
                        test_data, test_labels = data
                        #test_data = test_data.cuda()
                        y_pred = model(test_data)
                        y_pred = torch.sigmoid(y_pred)
                        test_pred.append(y_pred.cpu().detach().numpy())
                        test_true.append(test_labels.numpy())

                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    val_auc_mean = np.mean(auc_roc_score(test_true, test_pred))
                    model.train()
                    #print(val_auc_mean)
                    #if best_val_auc < val_auc_mean:
                    #    best_val_auc = val_auc_mean
                        #torch.save(model.state_dict(), 'aucm_pretrained_model.pth')

                    print('Epoch=%s, BatchID=%s, Val_AUC=%.4f' % (epoch, idx, val_auc_mean))

        '''
        # evaluation on train & test sets
        model.eval()
        train_pred_list = []
        train_true_list = []
        for train_data, train_targets in trainloader_eval:
            train_pred = model(train_data)
            train_pred = torch.sigmoid(train_pred)
            train_pred_list.append(train_pred.detach().numpy())
            train_true_list.append(train_targets.numpy())
        train_true = np.concatenate(train_true_list)
        train_pred = np.concatenate(train_pred_list)
        train_auc = auc_roc_score(train_true, train_pred)
        train_loss = np.mean(train_loss)

        val_pred_list = []
        val_true_list = []
        for val_data, val_targets in valloader:
            val_pred = model(val_data)
            val_pred = torch.sigmoid(val_pred)
            print(val_pred)
            val_pred_list.append(val_pred.detach().numpy())
            val_true_list.append(val_targets.numpy())
        val_true = np.concatenate(val_true_list)
        val_pred = np.concatenate(val_pred_list)
        val_auc = auc_roc_score(val_true, val_pred)
        model.train()

        print(train_loss,"\n", train_auc,"\n", val_auc,"\n", optimizer.lr)
        # print results
        print("epoch: %s, train_auc: %.4f, val_auc: %.4f, lr: %.4f" % (epoch,
                                                    train_auc, val_auc, optimizer.lr))
        train_log.append(train_auc)
        val_log.append(val_auc)
        '''
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        # early_stopping(valid_auc, model)
        #if val_auc <= last_val_auc or abs(val_auc - last_val_auc) <= 1e-3 * 5:
        if val_auc_mean <= last_val_auc or abs(val_auc_mean - last_val_auc) <= 1e-3 * 5:
            # validation auc decreasing or the difference of validation auc between epochs indicate the stopping point
            no_val_change_count += 1
        else:
            no_val_change_count = 0

        print(no_val_change_count)
        if no_val_change_count >= patience:
            print("Early stopping")
            break
        last_val_auc = val_auc_mean
        #last_val_auc = val_auc
    #return train_log, val_log
