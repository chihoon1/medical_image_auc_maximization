from sklearn import metrics
from tqdm import tqdm
import copy
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
import sys


# Adds Gaussian noise (mean, std) to a tensor. Works as a transform or called directly.
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



# Trainer for LibAUC losses (AUCMLoss, AUCMLoss_V2) with PESG. The model output goes through a sigmoid before loss_fn.
# The optimizer needs an `lr` attribute (printed each epoch) and, when decay_epochs is set, update_regularizer(), so use PESG.
# For a regular torch optimizer with a loss on raw logits, use train_with_logits_loss.
def train(model, loss_fn, optimizer, total_epochs, trainloader, trainloader_eval, testloader,
          patience=15, decay_epochs=None):
    # param: model = pytorch NN model, moved to the GPU when there is one
    # param: loss_fn = called as loss_fn(sigmoid(logits), targets)
    # param: optimizer = LibAUC PESG
    # param: total_epochs(int) = maximum number of epochs
    # param: trainloader = training batches
    # param: trainloader_eval = loader used for the train AUC printed each epoch
    # param: testloader = loader for the per-epoch validation ROC-AUC (pass the validation loader). It drives early stopping and the checkpoint.
    # param: patience = stop after this many epochs in a row without a gain (the AUC fell, or rose by at most 0.005 since the previous epoch)
    # param: decay_epochs = epochs at which the lr is divided by 10 (None = no decay)
    # returns (train_log, val_log), the per-epoch train and validation ROC-AUC. The best-validation weights are loaded back
    # into the model, which is left in train mode.

    print('Start Training')
    print('-' * 30)

    # AUCMLoss keeps its parameters (a, b) on cuda when available, so model and batches go to the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model.to(device)

    train_log = []
    test_log = []

    no_val_change_count = 0
    last_val_auc = 0
    # best-validation-AUC weights, loaded back after training
    best_val_auc = -1.0
    best_epoch = -1
    best_state = None
    for epoch in range(total_epochs):
        if decay_epochs is not None and epoch in decay_epochs:
            optimizer.update_regularizer(decay_factor=10)  # lr / 10 and PESG regularizer update

        train_loss = []
        model.train()
        for data, targets in trainloader:
            data, targets = data.to(device), targets.to(device)
            y_pred = model(data)
            y_pred = torch.sigmoid(y_pred)
            loss = loss_fn(y_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # train and validation ROC-AUC for this epoch
        model.eval()
        train_pred_list = []
        train_true_list = []
        for train_data, train_targets in trainloader_eval:
            train_data = train_data.to(device)
            train_pred = model(train_data)
            train_pred = torch.sigmoid(train_pred)
            train_pred_list.append(train_pred.detach().cpu().numpy())
            train_true_list.append(train_targets.numpy())
        train_true = np.concatenate(train_true_list)
        train_pred = np.concatenate(train_pred_list)
        train_auc = np.mean(auc_roc_score(train_true, train_pred))
        train_loss = np.mean(train_loss)

        test_pred_list = []
        test_true_list = []
        for test_data, test_targets in testloader:
            test_data = test_data.to(device)
            test_pred = model(test_data)
            test_pred = torch.sigmoid(test_pred)
            test_pred_list.append(test_pred.detach().cpu().numpy())
            test_true_list.append(test_targets.numpy())
        test_true = np.concatenate(test_true_list)
        test_pred = np.concatenate(test_pred_list)
        val_auc = np.mean(auc_roc_score(test_true, test_pred))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        model.train()

        print("epoch: %s, train_loss: %.4f, train_auc: %.4f, val_auc: %.4f, lr: %.4f" % (
        epoch, train_loss, train_auc, val_auc, optimizer.lr))
        train_log.append(train_auc)
        test_log.append(val_auc)

        if val_auc <= last_val_auc  or abs(val_auc - last_val_auc) <= 1e-3*5:
            # no gain: the AUC fell, or rose by at most 0.005 since the previous epoch
            no_val_change_count += 1
        else:
            no_val_change_count = 0

        print(no_val_change_count)
        if no_val_change_count >= patience:
            print("Early stopping")
            break

        last_val_auc = val_auc

    if best_state is not None:
        print("Restoring best checkpoint: epoch %d, val_auc %.4f" % (best_epoch, best_val_auc))
        model.load_state_dict(best_state)
    return train_log, test_log


def multilabel_train(model, loss_fn, optimizer, total_epochs, trainloader, trainloader_eval, valloader,
          patience=12, decay_epochs=None, min_delta=1e-3 * 5):
    # Trainer for LibAUC multi-label losses such as MultiLabelAUCMLoss with PESG. The model output goes through a sigmoid before loss_fn.
    # Validation runs at batch 0 and every 400th batch of an epoch (once per epoch when the loader has 400 batches or fewer),
    # so it happens after the first step of the epoch. It reports the macro ROC-AUC, which drives early stopping and the checkpoint.
    # Returns None. The best-validation weights are loaded back into the model, which is left in train mode.
    # param: model = pytorch NN model, moved to the GPU when there is one
    # param: loss_fn = called as loss_fn(sigmoid(logits), labels)
    # param: optimizer = LibAUC PESG (update_regularizer() is called when decay_epochs is set)
    # param: total_epochs(int) = maximum number of epochs
    # param: trainloader = training batches
    # param: trainloader_eval = not used, kept so the signature matches train()
    # param: valloader = validation batches
    # param: patience = stop after this many epochs in a row without a gain
    # param: decay_epochs = epochs at which the lr is divided by 10 (None = no decay)
    # param: min_delta = an epoch is "no gain" when the validation AUC fell, or rose by at most this much since the previous epoch
    #        (default 0.005). train_with_logits_loss takes the same min_delta; train() fixes it at 0.005.

    print('Start Training')
    print('-' * 30)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model.to(device)

    train_log = []
    val_log = []

    no_val_change_count = 0
    last_val_auc = 0
    # best-validation-AUC weights (checked at batch 0 and every 400th batch), loaded back at the end
    best_val_auc = -1.0
    best_epoch = -1
    best_state = None
    for epoch in range(total_epochs):
        if decay_epochs is not None and epoch in decay_epochs:
            optimizer.update_regularizer(decay_factor=10)  # lr / 10 and PESG regularizer update

        train_loss = []
        model.train()
        for idx, data in enumerate(trainloader):
            train_data, train_labels = data
            train_data, train_labels = train_data.to(device), train_labels.to(device)
            y_pred = model(train_data)
            y_pred = torch.sigmoid(y_pred)
            loss = loss_fn(y_pred, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            # validation at batch 0 and every 400th batch
            if idx % 400 == 0:
                model.eval()
                with torch.no_grad():
                    test_pred = []
                    test_true = []
                    for jdx, data in enumerate(valloader):
                        test_data, test_labels = data
                        test_data = test_data.to(device)
                        y_pred = model(test_data)
                        y_pred = torch.sigmoid(y_pred)
                        test_pred.append(y_pred.cpu().detach().numpy())
                        test_true.append(test_labels.numpy())

                    test_true = np.concatenate(test_true)
                    test_pred = np.concatenate(test_pred)
                    val_auc_mean = np.mean(auc_roc_score(test_true, test_pred))
                    if val_auc_mean > best_val_auc:
                        best_val_auc = val_auc_mean
                        best_epoch = epoch
                        best_state = copy.deepcopy(model.state_dict())
                    model.train()

                    print('Epoch=%s, BatchID=%s, Val_AUC=%.4f' % (epoch, idx, val_auc_mean))

        if val_auc_mean <= last_val_auc or abs(val_auc_mean - last_val_auc) <= min_delta:
            # no gain: the AUC fell, or rose by at most min_delta since the previous epoch
            no_val_change_count += 1
        else:
            no_val_change_count = 0

        print(no_val_change_count)
        if no_val_change_count >= patience:
            print("Early stopping")
            break
        last_val_auc = val_auc_mean

    if best_state is not None:
        print("Restoring best checkpoint: epoch %d, val_auc %.4f" % (best_epoch, best_val_auc))
        model.load_state_dict(best_state)


def train_with_logits_loss(model, loss_fn, optimizer, total_epochs, trainloader, trainloader_eval, valloader,
                           patience=12, min_delta=1e-3 * 5):
    # Trainer for binary (one output column) and multi-label (several columns) classification with a loss on raw logits.
    # loss_fn must return a scalar (BCEWithLogitsLoss, or sigmoid_focal_loss with reduction='mean'). The optimizer is a regular
    # torch one such as Adam. PESG is not supported (no decay_epochs). Each output column is its own binary label.
    # Validation uses sigmoid(logits) and reports the macro ROC-AUC (mean of the per-column AUCs), which drives early stopping
    # and the checkpoint. It runs at batch 0 and every 400th batch (once per epoch for 400 batches or fewer).
    # Returns None. The best-validation weights are loaded back into the model, which is left in train mode.
    # param: model = pytorch NN model, moved to the GPU when there is one
    # param: loss_fn = called as loss_fn(logits, labels.float()), must return a scalar
    # param: optimizer = regular torch optimizer (e.g. torch.optim.Adam)
    # param: total_epochs(int) = maximum number of epochs
    # param: trainloader = training batches
    # param: trainloader_eval = not used, kept so the signature matches train()
    # param: valloader = validation batches
    # param: patience = stop after this many epochs in a row without a gain
    # param: min_delta = an epoch is "no gain" when the validation AUC fell, or rose by at most this much since the previous epoch
    #        (default 0.005)

    print('Start Training')
    print('-' * 30)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model.to(device)

    no_val_change_count = 0
    last_val_auc = 0
    val_auc_mean = 0
    # best-validation-AUC weights, loaded back at the end
    best_val_auc = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(total_epochs):
        model.train()
        for idx, batch in enumerate(trainloader):
            train_data, train_labels = batch
            train_data, train_labels = train_data.to(device), train_labels.to(device)
            y_pred = model(train_data)
            loss = loss_fn(y_pred, train_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # validation at batch 0 and every 400th batch
            if idx % 400 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = []
                    val_true = []
                    for v_data, v_labels in valloader:
                        v_data = v_data.to(device)
                        v_pred = torch.sigmoid(model(v_data))
                        val_pred.append(v_pred.cpu().numpy())
                        val_true.append(v_labels.numpy())

                    val_true = np.concatenate(val_true)
                    val_pred = np.concatenate(val_pred)
                    val_auc_mean = np.mean(auc_roc_score(val_true, val_pred))
                    if val_auc_mean > best_val_auc:
                        best_val_auc = val_auc_mean
                        best_epoch = epoch
                        best_state = copy.deepcopy(model.state_dict())
                    model.train()

                    print('Epoch=%s, BatchID=%s, Val_AUC=%.4f' % (epoch, idx, val_auc_mean))

        # same early-stopping rule as multilabel_train
        if val_auc_mean <= last_val_auc or abs(val_auc_mean - last_val_auc) <= min_delta:
            no_val_change_count += 1
        else:
            no_val_change_count = 0

        print(no_val_change_count)
        if no_val_change_count >= patience:
            print("Early stopping")
            break
        last_val_auc = val_auc_mean

    if best_state is not None:
        print("Restoring best checkpoint: epoch %d, val_auc %.4f" % (best_epoch, best_val_auc))
        model.load_state_dict(best_state)


def build_weighted_sampler(labels, beta=0.5, class_weights=None, num_samples=None, replacement=True):
    # WeightedRandomSampler with per-sample weight 1 / class_count**beta, so minority samples are drawn more often.
    # Nothing is copied or dropped.
    # param: labels = 1-D array of single-label class values (binary or multi-class). Multi-label data is not supported:
    #        build per-sample weights yourself and pass them to WeightedRandomSampler.
    # param: beta = 1.0 gives equal sampling mass per class, 0.0 the natural distribution, 0.5 square-root tempering
    # param: class_weights = optional per-class weights that replace the beta formula: a dict {class_value: weight}
    #        or a list aligned with sorted(unique(labels)), e.g. class_weights={0: 1.0, 1: 3.0}
    # param: num_samples = draws per epoch (default len(labels))
    # param: replacement = passed to WeightedRandomSampler
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    count_by_class = dict(zip(classes.tolist(), counts.tolist()))
    if class_weights is None:
        weight_by_class = {c: 1.0 / count_by_class[c] ** beta for c in count_by_class}
    elif isinstance(class_weights, dict):
        weight_by_class = class_weights
    else:
        assert len(class_weights) == len(classes), "class_weights must have one entry per class"
        weight_by_class = dict(zip(classes.tolist(), class_weights))
    weights = np.array([weight_by_class[int(l)] for l in labels], dtype=np.float64)
    weights = torch.from_numpy(weights)
    if num_samples is None:
        num_samples = len(labels)
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=num_samples, replacement=replacement)


class OnTheFlyAugment(torch.utils.data.Dataset):
    # Wraps a dataset and applies a fresh random `augment` to every item on every access, whatever its class.
    # Augmenting only one class would let the model use the augmentation itself as a shortcut for that label.
    # numpy labels are converted to tensors.
    def __init__(self, base_dataset, augment):
        self.base = base_dataset
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
        return self.augment(img), label


def evaluate_val_auc(model, valloader, multi_label=False, metric='roc_auc'):
    # Scores the model on valloader with sigmoid(logits) and returns one number. The model is moved to the GPU when there
    # is one and left in eval mode.
    # param: multi_label = False gives one binary score, True averages over the label columns (macro)
    # param: metric = 'roc_auc' or 'pr_auc' (average precision). PR-AUC treats label 1 as the positive class, so it is a
    #        minority-class metric only when label 1 is the minority.
    # Handy for an Optuna score_fn when the tuning metric is not the trainer's own validation ROC-AUC.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    score_list, label_list = [], []
    with torch.no_grad():
        for tmp_data, tmp_label in valloader:
            tmp_data = tmp_data.to(device)
            tmp_score = torch.sigmoid(model(tmp_data)).detach().clone().cpu()
            score_list.append(tmp_score)
            label_list.append(tmp_label.cpu())
    val_label = torch.cat(label_list)
    val_score = torch.cat(score_list)
    if metric == 'roc_auc':
        if multi_label:
            return metrics.roc_auc_score(val_label, val_score, average='macro')
        return metrics.roc_auc_score(val_label, val_score)
    if metric == 'pr_auc':
        val_label, val_score = val_label.numpy(), val_score.numpy()
        if multi_label:
            return metrics.average_precision_score(val_label, val_score, average='macro')
        return metrics.average_precision_score(val_label.reshape(-1), val_score.reshape(-1))
    # NotImplementedError: a ValueError would be swallowed by run_optuna_search's catch=(ValueError,)
    # and every trial would fail silently
    raise NotImplementedError("metric must be 'roc_auc' or 'pr_auc', got %r" % (metric,))
