#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import logging
import numpy as np
from sklearn import preprocessing
from utils.data_provider import read_GD_dataset, read_HSS_dataset, read_S5_dataset, read_NAB_dataset, read_2D_dataset,read_ECG_dataset, get_loader, generate_synthetic_dataset, read_SMD_dataset, read_SMAP_dataset, read_MSL_dataset,rolling_window_2D, cutting_window_2D, unroll_window_3D, read_WADI_dataset, read_SWAT_dataset
from utils.metrics_insert import CalculateMetrics


# In[2]:


def ProcessResults(data_file,dataset,results_file,model_name,pid,file_name,rolling_size):
    train_data = None
    if dataset == 31 or dataset == 32 or dataset == 33 or dataset == 34 or dataset == 35:
        abnormal_data, abnormal_label = read_S5_dataset(data_file)
    if dataset == 41 or dataset == 42 or dataset == 43 or dataset == 44 or dataset == 45 or dataset == 46:
        abnormal_data, abnormal_label = read_NAB_dataset(data_file)
    if dataset == 51 or dataset == 52 or dataset == 53 or dataset == 54 or dataset == 55 or dataset == 56 or dataset == 57:
        train_data, abnormal_data, abnormal_label = read_2D_dataset(data_file)
    if dataset == 61 or dataset == 62 or dataset == 63 or dataset == 64 or dataset == 65 or dataset == 66 or dataset == 67:
        abnormal_data, abnormal_label = read_ECG_dataset(data_file)
    if dataset == 71 or dataset == 72 or dataset == 73:
        train_data, abnormal_data, abnormal_label = read_SMD_dataset(data_file)
    if dataset == 81 or dataset == 82 or dataset == 83 or dataset == 84 or dataset == 85 or dataset == 86 or dataset == 87 or dataset == 88 or dataset == 89 or dataset == 90:
        train_data, abnormal_data, abnormal_label = read_SMAP_dataset(data_file)
    if dataset == 91 or dataset == 92 or dataset == 93 or dataset == 94 or dataset == 95 or dataset == 96 or dataset == 97:
        train_data, abnormal_data, abnormal_label = read_MSL_dataset(data_file)
    if dataset == 101:
        train_data, abnormal_data, abnormal_label = read_SWAT_dataset(data_file, sampling=0.1)
    if dataset == 102 or dataset == 103:
        train_data, abnormal_data, abnormal_label = read_WADI_dataset(data_file, sampling=0.1)

    original_x_dim = abnormal_data.shape[1]
    
    dec_mean_unroll = np.load(results_file)
    
    if model_name != 'MAS':
        use_preprocessing = True
        use_overlapping = True
        use_last_point = True
    else:
        use_preprocessing = False
        use_overlapping = False
        use_last_point = False
    
    if use_preprocessing:
        if use_overlapping:
            if use_last_point:
                x_original_unroll = abnormal_data[rolling_size-1:]
                abnormal_segment = abnormal_label[rolling_size-1:]
            else:
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
        else:
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
            abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
    else:
        x_original_unroll = abnormal_data
        abnormal_segment = abnormal_label

    scaler = preprocessing.StandardScaler()
    x_original_unroll = scaler.fit_transform(x_original_unroll)

    if model_name[-8:] == 'Decision':
        error = np.load(results_file.replace('Decision','Score'))
    elif model_name == 'MAS':
        error = dec_mean_unroll
    else:
        error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]), axis=1) ** 2
    
    pos_label = -1
    score = error

    std_dev = [2,2.5,3]
    quantile = [97,98,99]
    k_values = [0.5,1,2,4,5,10]

    CalculateMetrics(abnormal_segment, score, pos_label, std_dev, quantile, k_values, model_name, pid, dataset, file_name)


# In[3]:


def get_dataset_path(dataset_number):
    if dataset_number == 31:
        path = './data/YAHOO/data/A1Benchmark'
    if dataset_number == 32:
        path = './data/YAHOO/data/A2Benchmark'
    if dataset_number == 33:
        path = './data/YAHOO/data/A3Benchmark'
    if dataset_number == 34:
        path = './data/YAHOO/data/A4Benchmark'
    if dataset_number == 35:
        path = './data/YAHOO/data/Vis'
    if dataset_number == 61:
        path = './data/ECG/chf01'
    if dataset_number == 62:
        path = './data/ECG/chf13'
    if dataset_number == 63:
        path = './data/ECG/ltstdb43'
    if dataset_number == 64:
        path = './data/ECG/ltstdb240'
    if dataset_number == 65:
        path = './data/ECG/mitdb180'
    if dataset_number == 66:
        path = './data/ECG/stdb308'
    if dataset_number == 67:
        path = './data/ECG/xmitdb108'
    if dataset_number == 71:
        path = './data/SMD/machine1/train'
    if dataset_number == 72:
        path = './data/SMD/machine2/train'
    if dataset_number == 73:
        path = './data/SMD/machine3/train'
    if dataset_number == 81:
        path = './data/SMAP/channel1/train'
    if dataset_number == 82:
        path = './data/SMAP/channel2/train'
    if dataset_number == 83:
        path = './data/SMAP/channel3/train'
    if dataset_number == 84:
        path = './data/SMAP/channel4/train'
    if dataset_number == 85:
        path = './data/SMAP/channel5/train'
    if dataset_number == 86:
        path = './data/SMAP/channel6/train'
    if dataset_number == 87:
        path = './data/SMAP/channel7/train'
    if dataset_number == 88:
        path = './data/SMAP/channel8/train'
    if dataset_number == 89:
        path = './data/SMAP/channel9/train'
    if dataset_number == 90:
        path = './data/SMAP/channel10/train'
    if dataset_number == 91:
        path = './data/MSL/channel1/train'
    if dataset_number == 92:
        path = './data/MSL/channel2/train'
    if dataset_number == 93:
        path = './data/MSL/channel3/train'
    if dataset_number == 94:
        path = './data/MSL/channel4/train'
    if dataset_number == 95:
        path = './data/MSL/channel5/train'
    if dataset_number == 96:
        path = './data/MSL/channel6/train'
    if dataset_number == 97:
        path = './data/MSL/channel7/train'
    if dataset_number == 101:
        path = './data/SWaT/train'
    if dataset_number == 102:
        path = './data/WADI/2017/train'
    if dataset_number == 103:
        path = './data/WADI/2019/train'
        
    return path


# In[4]:


if __name__ == '__main__':
    path = './save_outputs/NPY/'
    logging.basicConfig(level=logging.DEBUG, filename="./Log.txt", filemode="a+",format="%(message)s")
    
    for root, dirs, files in os.walk(path):
        if root != path:
            dataset = int(root.split('/')[-1])
            
            for file in files:
                if (file[0:3] != 'Dec'):
                    parsed = file.split('#')
                    model = parsed[0]
                    if model[-5:] != 'Score':
                        pid = int(parsed[2])
                        file_name = parsed[1]
                        results_file = os.path.join(root, file)
                        dataset_path = get_dataset_path(dataset)
                        prefixed = [filename for filename in os.listdir(dataset_path) if filename.startswith(file_name)]
                        data_file = os.path.join(dataset_path, prefixed[0])
                        rolling_size = int(parsed[3][:-4])
                        try:
                             ProcessResults(data_file,dataset,results_file,model,pid,file_name,rolling_size)
                        except Exception as e:
                             logging.info("Error with " + results_file + " Message: " + str(e))
                
                elif (file[0:3]=='Dec'):
                    model = file.split('_')[1]
                    if model == 'RNNVAE' or model == 'OMNIANOMALY':
                        pid = int(file.split('_')[-1].split('=')[-1][0:-4])
                        file_name = '_'.join(file.split('_')[2:-1])
                        results_file = os.path.join(root, file)
                        dataset_path = get_dataset_path(dataset)
                        prefixed = [filename for filename in os.listdir(dataset_path) if filename.startswith(file_name)]
                        data_file = os.path.join(dataset_path, prefixed[0])
                        
                        if dataset in (71,72,73,102,103):
                            rolling_size = 32
                        else:
                            rolling_size = 16
                        
                        if model == 'OMNIANOMALY':
                            rolling_size = 100
                        
                        try:
                            ProcessResults(data_file,dataset,results_file,model,pid,file_name,rolling_size)
                        except Exception as e:
                            logging.info("Error with " + results_file + " Message: " + str(e))

