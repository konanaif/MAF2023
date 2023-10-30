import pandas as pd
import numpy as np
import os
from collections import Counter

from aif360.datasets import StandardDataset

import torch
from torch.utils.data import Dataset


class aifData(StandardDataset):
    def __init__(self, df, label_name, favorable_classes,
                 protected_attribute_names, privileged_classes,
                 instance_weights_name='', scores_name='',
                 categorical_features=[], features_to_keep=[],
                 features_to_drop=[], na_values=[], custom_preprocessing=None,
                 metadata=None):
        
        super(aifData, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)



class RawDataSet:
    def __init__(self, filename=None, **kwargs):
        if filename:
            # Detect file type
            extention = os.path.splitext(filename)[-1]
        else:
            extention = ''

        ##### CASE 1 : numpy object #####
        if extention == '.npy':
            ## Required parameters
            #    - target column index (int) : target_col_idx
            #    - bias column index (int) : bias_col_idx
            ## Optional parameters
            #    - prediction column index (int) : pred_col_idx

            # Check required parameters
            try:
                target_col_idx = kwargs['target_col_idx']
                bias_col_idx = kwargs['bias_col_idx']
            except:
                print('ERROR! You need to pass the required parameters.')
                print('Please check the [target_col_idx] and [bias_col_idx].')
                raise

            # Load the raw file
            loaded_arr = np.load(filename)

            # Make features
            self.target = loaded_arr[:, target_col_idx]
            self.bias = loaded_arr[:, bias_col_idx]

            # Check optional parameters
            if 'pred_col_idx' in kwargs.keys():
                pred_col_idx = kwargs['pred_col_idx']
                self.predict = loaded_arr[:, pred_col_idx]
                self.feature = np.delete(loaded_arr, [target_col_idx, pred_col_idx], axis=1)
                self.feature_only = np.delete(loaded_arr, [bias_col_idx, target_col_idx, pred_col_idx], axis=1)
            else:
                self.predict = np.zeros_like(loaded_arr[:, 0]) - 1  # [-1, -1, -1, ..., -1]
                self.feature = np.delete(loaded_arr, [target_col_idx], axis=1)
                self.feature_only = np.delete(loaded_arr, [bias_col_idx, target_col_idx], axis=1)
        #################################

        ##### CASE 2 : framed table #####
        elif extention in ['.csv', '.tsv']:
            ## Required parameters
            #    - target column name (str) : target_col_name
            #    - bias column name (str) : bias_col_name
            ## Optional parameters
            #    - header (int) : table header (column name) row index
            #    - seperator (str) : table seperator
            #    - prediction column name (str) : pred_col_name
            #    - categorical column name list (list) : cate_cols

            # Check the required parameters
            try:
                target_col_name = kwargs['target_col_name']
                bias_col_name = kwargs['bias_col_name']
            except:
                print('ERROR! You need to pass the required parameters.')
                print('Please check the [target_col_idx] and [bias_col_idx].')
                raise

            # Check the optional parameters
            header = kwargs['header'] if 'header' in kwargs.keys() else 0             # default 0
            seperator = kwargs['seperator'] if 'seperator' in kwargs.keys() else ','  # default ,
            cate_cols = kwargs['cate_cols'] if 'cate_cols' in kwargs.keys() else []   # default []

            # Load the table file
            loaded_df = pd.read_table(filename, sep=seperator, header=header)

            # Preprocess categorical columns
            converted_df = self.convert_categorical(loaded_df, cate_cols)


            # Make features
            self.target = converted_df.loc[:, target_col_name].to_numpy()
            self.bias = converted_df.loc[:, bias_col_name].to_numpy()
            if 'pred_col_name' in kwargs.keys():
                pred_col_name = kwargs['pred_col_name']
                self.predict = converted_df.loc[:, pred_col_name].to_numpy()
                self.feature = converted_df.drop(columns=[target_col_name, pred_col_name]).to_numpy()
                self.feature_only = converted_df.drop(columns=[bias_col_name, target_col_name, pred_col_name]).to_numpy()
            else:
                self.predict = np.zeros_like(converted_df[target_col_name]) - 1  # [-1, -1, -1, ..., -1]
                self.feature = converted_df.drop(columns=[target_col_name]).to_numpy()
                self.feature_only = converted_df.drop(columns=[bias_col_name, target_col_name]).to_numpy()

        else:
            try:
                self.feature = kwargs['x']
                self.bias = kwargs['z']
                self.target = kwargs['y']
                self.feature_only = kwargs['x']
            except:
                print("Input file : {}\t\t\tExtention : {}".format(filename, extention))
                raise Exception("FILE ERROR!! Only [npy, csv, tsv] extention required.")


    # Convert categorical values to numerical (integer) values on pandas.DataFrame
    def convert_categorical(self, dataframe, category_list):
        temp = dataframe.copy()

        for cate in category_list:
            categories = Counter(temp[cate])

            c2i = {}
            i = 0
            for c, f in categories.items():
                #i = i + 1
                c2i[c] = i

            temp[cate] = temp[cate].map(lambda x: c2i[x])

        return temp



class PytorchDataset(Dataset):
    def __init__(self, kaif_rawdata):
        self.feature = kaif_rawdata.feature
        self.bias = kaif_rawdata.bias
        self.target = kaif_rawdata.target
        self.predict = kaif_rawdata.predict

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return feature[idx], target[idx]