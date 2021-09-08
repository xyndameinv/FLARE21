# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:12:25 2019

@author: Tan
"""

import os
import random
import numpy as np
from skimage import measure


def get_binary_xy(in_file, dim_num=2):
    '''
    获取二维二值矩阵的轮廓xy-plane
    in_file:为输入的矩阵
    return:contour的坐标
    '''
    binary_xy = []
    if dim_num > 2:
        print('the input must be two dims.')
        exit()
    contours = measure.find_contours(in_file, 0.5)
    contour = contours[0]
    contour = np.around(contour, 1).astype(np.int)
    #    contour[:, [0, 1]] = contour[:, [1, 0]]

    for i in range(contour.shape[0]):
        binary_xy.append(list(contour[i, :]))

    return binary_xy


def get_target_range(in_file):
    '''
    获取in_file矩阵里,最大非0边界范围
    in_file:为输入的矩阵
    return:为范围的min/max
    '''
    get_binar = np.where(in_file == 1)

    w, h, d = (get_binar[0], get_binar[1], get_binar[2])
    #    max+1是因为在in_file[w_min:w_max,h_min:h_max,d_min:d_max]，
    #   不会取到max,需要+1，确保所有label==1的取完整
    w_min, w_max = (w.min(), w.max() + 1)
    h_min, h_max = (h.min(), h.max() + 1)
    d_min, d_max = (d.min(), d.max() + 1)
    return w_min, w_max, h_min, h_max, d_min, d_max


def get_binary_coors(in_file, dim_num=2):
    '''
    获取三维的轮廓坐标
    in_file:为输入的矩阵（label）
    return:binary_coor_list为轮廓坐标[(w0,h0,d0),(w1,h1,d1),...]
            center_contour_img是输入对应的骨架线图矩阵
    '''
    center_contour_img = np.zeros(in_file.shape)
    w_min, w_max, h_min, h_max, d_min, d_max = get_target_range(in_file)
    usege_file = in_file[:, :, d_min:d_max]
    binary_coors = []
    for i in range(usege_file.shape[2]):
        binary_xy = get_binary_xy(usege_file[:, :, i], dim_num=2)
        for j in range(len(binary_xy)):
            binary_xy[j].append(i + d_min)
            binary_coors.extend(binary_xy)

    for i in range(len(binary_coors)):
        xyz = binary_coors[i]
        center_contour_img[xyz[0], xyz[1], xyz[2]] = 1.0

    binary = np.where(center_contour_img)
    binary_coor_list = []
    for k in range(binary[0].shape[0]):
        coor = (binary[0][k], binary[1][k], binary[2][k])
        binary_coor_list.append(coor)

    return binary_coor_list, center_contour_img


def get_seeds_list(point_range_list, seed_num):
    '''
    获取在point_range_list间,seed_num个点
    point_range_list:是取点的范围，定义为一个list,如果是[0,10]和[20,30]
    之间，可以采用list(range(0,11))+list(range(20,31))形成的point_range_list
    '''
    seed_point_list = []
    for i in range(0, seed_num):
        seed_index = random.randint(0, len(point_range_list) - 1)
        seed_point_list.append(point_range_list[seed_index])
    return seed_point_list


def get_seed_coordinate(in_file, seed_num, positive_sample=True):
    '''
    in_file：待切块的矩阵
    seed_num：需要的撒点数
    positive_sample：是否是正样本，如果True,在[min,max)之间取值，否则在[0,min)和(max,in_file.shape)之间取值
    return：为种子的坐标，如[(w0,h0,d0),(w1,h1,d1),...]
    '''
    w_min, w_max, h_min, h_max, d_min, d_max = get_target_range(in_file)

    if positive_sample:
        w_range = list(range(w_min, w_max))
        h_range = list(range(h_min, h_max))
        d_range = list(range(d_min, d_max))
    else:
        w_range = list(range(0, w_min)) + list(range(w_max, in_file.shape[0]))
        h_range = list(range(0, h_min)) + list(range(h_max, in_file.shape[1]))
        d_range = list(range(0, d_min)) + list(range(d_max, in_file.shape[2]))
        #排除每个Slice都有目标情况下，len（d_range）=0
        if len(d_range)==0:
            d_range = list(range(in_file.shape[2]))

    w_seed = get_seeds_list(w_range, seed_num)
    h_seed = get_seeds_list(h_range, seed_num)
    d_seed = get_seeds_list(d_range, seed_num)

    seeds_coor_list = []
    for i in range(0, seed_num):
        seed_coor = (w_seed[i], h_seed[i], d_seed[i])
        seeds_coor_list.append(seed_coor)

    return seeds_coor_list


def random_cut_patch(truth, patch_shape, binary_num_rate=0.025,min_point=900,
                     pos_num_rate=0.5, nag_num_rate=0.5):
    '''
    truth:与image对应的label array
    binary_num_rate=0.25:int(是所取边界点数量/所有边界点数量),这样可根据目标体积大小选取块
    pos_num_rate,nag_num_rate:int(为正负种子点的数量/所取边界点的数量)
    patch_shape:为块的大小，默认(64,64,32)
    return：
    '''
    #print(truth.shape)
    binary_seeds_list, center_contour_img = get_binary_coors(truth, dim_num=2)
    nums = len(binary_seeds_list)
    if (nums * binary_num_rate) * (1 + pos_num_rate + nag_num_rate) < min_point:
        rate = ((nums * binary_num_rate) * (1 + pos_num_rate + nag_num_rate)) / min_point
        binary_num_rate = binary_num_rate / rate
    
    index_point = [i for i in range(nums)]
    binary_seed_num = int(nums * binary_num_rate)
    binary_seeds_coor = [binary_seeds_list[j] for j in random.sample(index_point, binary_seed_num)]

    pos_seed_num = int(binary_seed_num * pos_num_rate)
    nag_seed_num = int(binary_seed_num * nag_num_rate)
    pos_seeds_coor = get_seed_coordinate(truth, pos_seed_num, positive_sample=True)
    nag_seeds_coor = get_seed_coordinate(truth, nag_seed_num, positive_sample=False)
    pos_seeds_coor.extend(binary_seeds_coor)
    pos_seeds_coor.extend(nag_seeds_coor)

    dw, dh, dd = patch_shape
    start_point_list=[]
    for index in range(0, len(pos_seeds_coor)):
        w, h, d = pos_seeds_coor[index]
        w_min, w_max = (w - int(dw / 2), w + int(dw / 2))
        h_min, h_max = (h - int(dh / 2), h + int(dh / 2))
        d_min, d_max = (d - int(dd / 2), d + int(dd / 2))

        '''
        问题是在shape[2]<patch_volume[2]情况下，下面两种if会同时起效，
        以第二if语句为准时，同时不能保证max-min==32,会出现负数下标的情况，
        此时会出现不足32，甚至更小
        怎么简洁判断min/max是否超出边界,以下判断存在问题，比如max-min
        '''
        if w_min < 0:
            w_min, w_max = (0, patch_shape[0])
        if w_max > truth.shape[0]:
            w_min, w_max = (truth.shape[0] - patch_shape[0], truth.shape[0])
        if h_min < 0:
            h_min, h_max = (0, patch_shape[1])
        if h_max > truth.shape[1]:
            h_min, h_max = (truth.shape[1] - patch_shape[1], truth.shape[1])
        if d_min < 0:
            d_min, d_max = (0, patch_shape[2])
        if d_max > truth.shape[2]:
            d_min, d_max = (truth.shape[2] - patch_shape[2], truth.shape[2])
        
        start_point_list.append(tuple([w_min,h_min,d_min]))#切块的左上角的点+patch_shape即可
    del pos_seeds_coor
    return start_point_list