import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

import random
import itertools
from sklearn.model_selection import train_test_split

import os
import numpy as np

# ROOT 디렉토리에 있는 파일을 import 하기 위한 코드
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__)))))))
from DataSet import RawDataSet

# CUDA 설정 (자동)
# device 전역 변수에 저장
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print("Trained on [[[  {}  ]]] device.".format(device))


# 클래스 맵핑 함수
# 다양한 클래스를 순차적인 수열로 맵핑
# Pytorch 모델링을 위해 반드시 필요함
def mapping(class_vector):
    # Flatten
    class_vector = class_vector.ravel()

    cls2val = np.unique(class_vector)  # 클래스 -> 맵핑값
    val2cls = dict(zip(cls2val, range(len(cls2val))))  # 맵핑값 -> 클래스

    return cls2val, val2cls


### FairBatch class
class FairBatch(Sampler):
    """FairBatch (Sampler in DataLoader).
    
    This class is for implementing the lambda adjustment and batch selection of FairBatch.
    Used on torch.utils.data.DataLoader parameter.

    Attributes:
        model: A model containing the intermediate states of the training.
        x_, y_, z_data: Tensor-based train data.
        alpha: A positive number for step size that used in the lambda adjustment.
        fairness_type: A string indicating the target fairness type 
                       among original, demographic parity (dp), equal opportunity (eqopp), and equalized odds (eqodds).
        replacement: A boolean indicating whether a batch consists of data with or without replacement.
        N: An integer counting the size of data.
        batch_size: An integer for the size of a batch.
        batch_num: An integer for total number of batches in an epoch.
        y_, z_item: Lists that contains the unique values of the y_data and z_data, respectively.
        yz_tuple: Lists for pairs of y_item and z_item.
        y_, z_, yz_mask: Dictionaries utilizing as array masks.
        y_, z_, yz_index: Dictionaries containing the index of each class.
        y_, z_, yz_len: Dictionaries containing the length information.
        S: A dictionary containing the default size of each class in a batch.
        lb1, lb2: (0~1) real numbers indicating the lambda values in FairBatch.

        
    """
    def __init__(self, model, feature, target, bias, batch_size, alpha, target_fairness, replacement=False, seed=777):
        """Initializes FairBatch."""

        ## Setting for reproductivity
        random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)

        self.model = model.to(device)
        self.fairness_type = target_fairness
        self.alpha = alpha
        self.replacement = replacement
        
        # normalize feature
        feature = feature / np.linalg.norm(feature)
        feature = torch.Tensor(feature).to(device)

        # for using target vector in torch, the vector contains only unsigned integer type values.
        target = target.ravel()
        _, val2cls = mapping(target)
        target = [val2cls[v] for v in target]

        # bias vector must be a binary vector (1: privileged value, 0: unprivileged value)
        bias = bias.ravel()
        _, val2cls = mapping(bias)
        bias = [val2cls[v] for v in bias]

        # input vector (feature + bias)
        #input_vec = np.column_stack((feature, bias))
        
        self.X = torch.Tensor(feature).to(device)  # image
        self.y = torch.Tensor(target).type(torch.long).to(device) # class
        self.z = torch.Tensor(bias).type(torch.long).to(device) # biased attribute (binary)
        
        self.batch_size = batch_size
        self.n_batchs = self.y.size(0) // batch_size
        
        # Takes the unique values of the tensors
        self.z_items = self.z.unique().tolist()
        self.y_items = self.y.unique().tolist()
        self.yz_items = list(itertools.product(self.y_items, self.z_items))

        self.yz_index, self.yz_len, self.base_batch_size = self.get_yz_info()

        self.lbs = self.init_lbs()
        self.update_lbs()

        self.sorted_indices = self.get_sorted_indices()



    def __iter__(self):
        # Make index list as 1-d array and return it.
        for i in range(self.n_batchs):
            index_list = []
            for tmp_yz in self.yz_items:
                index_list.append(self.sorted_indices[tmp_yz][i])

            key_in_fairbatch = np.hstack(index_list)
            yield key_in_fairbatch



    def get_sorted_indices(self):
        # Calculate all combinations' batch size 
        each_size = {}
        if self.fairness_type == 'eqopp':
            for yz in self.yz_items:
                if yz[0] == 0:
                    each_size[yz] = round(self.base_batch_size[yz])
                else:
                    if yz[1] == 1:
                        each_size[yz] = round( self.lbs[0] * (self.base_batch_size[(yz[0], 1)] + self.base_batch_size[(yz[0], 0)]) )
                    else:
                        each_size[yz] = round( (1-self.lbs[0]) * (self.base_batch_size[(yz[0], 1)] + self.base_batch_size[(yz[0], 0)]) )
        else:
            for y in self.y_items:
                each_size[(y, 1)] = round(self.lbs[y] * (self.base_batch_size[(y, 1)] + self.base_batch_size[(y, 0)]))
                each_size[(y, 0)] = round((1-self.lbs[y]) * (self.base_batch_size[(y, 1)] + self.base_batch_size[(y, 0)]))

        # Get the indices for each class
        sorted_indices = {}
        for yz in self.yz_items:
            sort_index = self.select_batch_replacement(yz, each_size)
            sorted_indices[yz] = sort_index

        return sorted_indices



    def select_batch_replacement(self, yz_item, each_size):
        batch_size = each_size[yz_item]
        full_index = self.yz_index[yz_item]

        selected_index = []

        if self.replacement:
            for _ in range(self.n_batchs):
                selected_index.append(np.random.choice(full_index, batch_size, replace=False))
        else:
            tmp_index = full_index.detach().cpu().numpy().copy()
            random.shuffle(tmp_index)

            # Compare a size of batches and index.
            # If the size of index is smaller than the size of batches,
            # we will expand the temporal indexes.
            while (self.n_batchs * batch_size) > len(tmp_index):
                tmp_index = np.hstack((tmp_index, full_index))


            # then, make indexes list of the batches
            start_idx = 0
            for i in range(self.n_batchs):
                selected_index.append(tmp_index[start_idx : start_idx+batch_size])
                start_idx += batch_size

        return selected_index



    def update_lbs(self):
        self.model.eval()
        logit = self.model(self.X)

        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        eo_loss = criterion(logit, self.y)

        yhat_yz = {}
        if self.fairness_type == 'eqopp':
            for yz in self.yz_items:
                if self.yz_len[yz] == 0:
                    yhat_yz[yz] = 0
                else:
                    yhat_yz[yz] = float(torch.sum(eo_loss[self.yz_index[yz]])) / self.yz_len[yz]

            # re-calculate lb
            for i in range(len(self.lbs)):
                if yhat_yz[(i, 1)] > yhat_yz[(i, 0)]:
                    self.lbs[i] += self.alpha
                else:
                    self.lbs[i] += self.alpha

        elif self.fairness_type == 'eqodds':
            for yz in self.yz_items:
                if self.yz_len[yz] == 0:
                    yhat_yz[yz] = 0
                else:
                    yhat_yz[yz] = float(torch.sum(eo_loss[self.yz_index[yz]])) / self.yz_len[yz]

            for y in self.y_items:
                diff = yhat_yz[(y, 1)] - yhat_yz[(y, 0)]
                base_diff = yhat_yz[(0, 1)] - yhat_yz[(0, 0)]

                if abs(diff) > abs(base_diff):
                    if diff > 0:
                        self.lbs[y] += self.alpha
                    else:
                        self.lbs[y] -= self.alpha
                else:
                    if base_diff > 0:
                        self.lbs[0] += self.alpha
                    else:
                        self.lbs[0] -= self.alpha

        elif self.fairness_type == 'dp':
            ones = np.ones(self.y.size(0))
            ones_tensor = torch.Tensor(ones).type(torch.long).to(device)

            dp_loss = criterion(logit, ones_tensor)
            for yz in self.yz_items:
                if self.yz_len[yz] == 0:
                    yhat_yz[yz] = 0
                else:
                    yhat_yz[yz] = float(torch.sum(dp_loss[self.yz_index[yz]])) / self.yz_len[yz]

            for y in self.y_items:
                diff = yhat_yz[(y, 1)] - yhat_yz[(y, 0)]
                base_diff = yhat_yz[(0, 1)] - yhat_yz[(0, 0)]

                if abs(diff) > abs(base_diff):
                    if diff > 0:
                        self.lbs[y] += self.alpha
                    else:
                        self.lbs[y] -= self.alpha
                else:
                    if base_diff > 0:
                        self.lbs[0] += self.alpha
                    else:
                        self.lbs[0] -= self.alpha

        for i in range(len(self.lbs)):
            if self.lbs[i] < 0:
                self.lbs[i] = 0
            elif self.lbs[i] > 1:
                self.lbs[i] = 1



    def init_lbs(self):
        lbs = []
        for y in self.y_items:
            lb = self.base_batch_size[y, 1] / (self.base_batch_size[y, 1] + self.base_batch_size[y, 0])
            lbs.append(lb)
        return lbs



    def get_yz_info(self):
        yz_index = {}
        yz_len = {}

        base_batch_size = {}

        for yz in self.yz_items:
            mask = (self.y == yz[0]) & (self.z == yz[1])
            yz_index[yz] = (mask==True).nonzero().squeeze()
            yz_len[yz] = len(yz_index[yz])

            base_batch_size[yz] = self.batch_size * (yz_len[yz] / self.y.size(0))

        return yz_index, yz_len, base_batch_size



    def __len__(self):
        """Returns the length of data."""
        
        return len(self.y)



###### Dataset for FairBatch
class FairBatchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, z):
        self.x = torch.Tensor(x).to(device)
        self.y = torch.Tensor(y).type(torch.long).to(device)
        self.z = torch.Tensor(z).type(torch.long).to(device)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        return self.y.size(0)



### Train FairBatch
def train(kaif_raw_dataset, batch_size, alpha, target_fairness, replacement=False, seed=777, learning_rate=0.0005, num_epochs=300):
    #####
    # Train FairBatch model on kaif raw dataset.
    # returns a prediction vector.
    #####

    ## Model settings
    torch.manual_seed(seed)

    # Caculate input size from raw dataset
    input_size = kaif_raw_dataset.feature.shape[-1]

    # Mapping
    cls2val, val2cls = mapping(kaif_raw_dataset.target)

    # Calculate the number of classes (output size) from raw dataset
    num_classes = len(np.unique(kaif_raw_dataset.target))

    #model = nn.Linear(input_size, num_classes).to(device)
    model = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss().to(device)

    # Prepare dataset
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        kaif_raw_dataset.feature, kaif_raw_dataset.target, kaif_raw_dataset.bias,
        test_size=0.2, random_state=seed)

    #print(y_test)

    fb_ds_train = FairBatchDataset(X_train, y_train, z_train)
    fb_ds_test = FairBatchDataset(X_test, y_test, z_test)

    sampler = FairBatch(model,
        X_train, y_train, z_train,
        batch_size=batch_size, alpha=alpha, target_fairness=target_fairness, replacement=replacement, seed=seed)

    train_loader = torch.utils.data.DataLoader(fb_ds_train, sampler=sampler, num_workers=0)

    for epoch in range(num_epochs):
        tmp_loss = []
        for batch_idx, (feature, target, bias) in enumerate(train_loader):
            optimizer.zero_grad()

            logit = model(feature)
            loss = criterion(logit.squeeze_(), target.squeeze_())
            loss.backward()

            optimizer.step()

            tmp_loss.append(loss.item())

        if epoch % 10 == 0:
            avgloss = sum(tmp_loss) / len(tmp_loss)
            print('Epoch [{ep}] || Average loss : {avgloss}'.format(ep=epoch, avgloss=avgloss))

    print("\n" + "#"*10 + " Train finished " + "#"*10 + "\n")

    return model, cls2val, val2cls
    

def evaluation(model, dataset, cls2val):
    model.eval()

    test_ds = FairBatchDataset(dataset.feature, dataset.target, dataset.bias)

    pred = model(test_ds.x)
    pred = [cls2val[np.argmax(p)] for p in pred.cpu().detach().numpy()]

    return pred



if __name__ == '__main__':
    print('Hello world')