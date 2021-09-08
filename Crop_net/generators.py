# -*- coding: utf-8 -*-
"""
Created on Mon May  6 21:03:03 2019

@author: Tan
"""
import copy
from random import shuffle
import itertools

import numpy as np
from keras.utils import np_utils

from get_patch_from_train import random_cut_patch
from patches import compute_patch_indices,get_patch_from_3d_data

def get_training_and_validation_generators(train_data,val_data, patch_shape, batch_size, n_labels,binary_num_rate,
                                           min_point,pos_num_rate,nag_num_rate,validation_patch_overlap=0):
    """
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    training_list = list(range(len(train_data)))
    training_generator = data_generator(train_data, training_list,
                                        batch_size=batch_size,
                                        binary_num_rate=binary_num_rate,
                                        min_point=min_point,
                                        pos_num_rate=pos_num_rate,
                                        nag_num_rate=nag_num_rate,
                                        n_labels=n_labels,
                                        patch_shape=patch_shape,
                                        shuffle_index_list=True)
    num_training_steps = get_number_of_steps(get_number_of_patches(train_data, training_list, binary_num_rate,min_point,
                                                                   pos_num_rate, nag_num_rate, patch_shape=patch_shape,
                                                                   patch_overlap=0), batch_size)
    print("Number of training steps: ", num_training_steps)
    
    if len(val_data):
        validation_list = list(range(len(val_data)))
        validation_generator = data_generator(val_data, validation_list,
                                              batch_size=batch_size,
                                              n_labels=n_labels,
                                              patch_shape=patch_shape,
                                              patch_overlap=validation_patch_overlap)
    
        num_validation_steps = get_number_of_steps(get_number_of_patches(val_data, validation_list,binary_num_rate,min_point=0,
                                                                         pos_num_rate=0, nag_num_rate=0, patch_shape=patch_shape,
                                                                         patch_overlap=validation_patch_overlap),batch_size)
    else:
        validation_generator = None
        num_validation_steps = None
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps

def data_generator(data_file, index_list, batch_size,binary_num_rate=0,min_point=0,pos_num_rate=0, nag_num_rate=0,
                   n_labels=1,patch_shape=None,patch_overlap=0,shuffle_index_list=False):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        index_list = copy.copy(orig_index_list)

        if patch_shape:
            index_list = create_patch_index_list(data_file, index_list, patch_shape, patch_overlap,
                                                 binary_num_rate,min_point,pos_num_rate, nag_num_rate)

        if shuffle_index_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            data, truth = get_data_from_file(data_file, index, patch_shape=patch_shape)
            x_list.append(data)
            y_list.append(truth)

            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                #print('generator:::')
                #print(len(x_list))
                #print(len(y_list))
                x,y = convert_data(x_list, y_list, n_labels=n_labels)
                #print('after convert')
                #print(x.shape)
                #print(y.shape)
                yield x,y
                x_list = list()
                y_list = list()

def get_number_of_patches(data_file, index_list, binary_num_rate,min_point=0,pos_num_rate=0, nag_num_rate=0,
                          patch_shape=None, patch_overlap=0):
    if patch_shape:
        index_list = create_patch_index_list(data_file, index_list, patch_shape, patch_overlap,
                                             binary_num_rate,min_point,pos_num_rate, nag_num_rate)
    return len(index_list)

def create_patch_index_list(datas, index_list, patch_shape, patch_overlap,binary_num_rate=0,min_point=0,
                            pos_num_rate=0, nag_num_rate=0):
    patch_index = list()
    for index in index_list:
        if not patch_overlap:
            #训练数据采用随机撒点进行采样
            truth=datas[index][2][:,:,:,0]
            patches=random_cut_patch(truth,patch_shape=patch_shape,binary_num_rate=binary_num_rate,
                                     min_point=min_point,pos_num_rate=pos_num_rate,nag_num_rate=nag_num_rate)
        else:#对验证数据滑窗去坐标点
            patches = compute_patch_indices(datas[index][2][:,:,:,0].shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index

def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1
    
def get_data_from_file(data_file, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(data_file, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y = data_file[index][1], data_file[index][2]
    return x, y

def convert_data(x_list, y_list, n_labels=1):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels > 1:
        y = np_utils.to_categorical(y[...,0],n_labels)
    return x, y