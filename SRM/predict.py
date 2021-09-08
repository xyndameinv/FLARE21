# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:09:27 2021

@author: Administrator
"""
#验证数据resize 128*128*96、标准化（最开始的尺寸要记录）
#测试结果mask resize原尺寸
import numpy as np
import SimpleITK as sitk
from glob import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from training import load_old_model
import nibabel as nib
from skimage import morphology
import shutil,copy
from keras.utils import np_utils
def test_dice_coef(mask, pred,smooth=1.0):
    '''
    对预测后的npy ,在做了拼接为nii后进行单个的dice计算
    '''
    mask = mask.flatten()
    pred = pred.flatten()
    intersection = np.sum(mask * pred)
    dice = (2 * intersection + smooth) / (np.sum(mask) + np.sum(pred) + smooth)
    return dice

def mutil_dice(n_class,truth,pred):
    all_dice = dict()
    gt = copy.copy(truth)
    pre = copy.copy(pred)
    gt = np_utils.to_categorical(gt,n_class+1)
    pre = np_utils.to_categorical(pre, n_class+1)
    for i in range(1,n_class+1):
        all_dice[i] = round(test_dice_coef(gt[:,:,:,i],pre[:,:,:,i]),4)
    return all_dice
def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled
def normalize_data(data):
    # b = np.percentile(data, 98)
    # t = np.percentile(data, 1)
    # data = np.clip(data,t,b)
    data = np.array(data,dtype=np.float32)
    means = data.mean()
    stds = data.std()
    # print(type(data),type(means),type(stds))
    data -= means
    data /= stds
    return data
    
def main():
    names=os.listdir(r'H:\mutil_CT_seg\data\SRM\TestImg')
    model=load_old_model(r'H:\mutil_CT_seg\result_srm\unet3d\model_hdf5\model.07-0.01.hdf5')
    dice_file = open(os.path.join(r'H:\mutil_CT_seg\result_srm\unet3d', 'post_dice_close.txt'),'a')
    for name in names:
        print(name)
        img=sitk.ReadImage(os.path.join(r'H:\mutil_CT_seg\data\SRM\TestImg',name))
        mask=sitk.ReadImage(os.path.join(r'H:\mutil_CT_seg\data\SRM\TestMask',name))
        imgarr=sitk.GetArrayFromImage(img)
        maskarr=sitk.GetArrayFromImage(mask) #96 128 128
        swap_resize_imgarr=np.swapaxes(imgarr,0,2) #128 128 96
        #swap_resize_maskarr=np.swapaxes(maskarr,0,2) #128 128 96
        #将swap_resize_imgarr输入网络测试 得到prediction 计算此时的dice的值（与resize_maskarr计算）
        imgarr = np_utils.to_categorical(np.expand_dims(swap_resize_imgarr, axis=-1)[...,0],5).astype(np.float32)
        imgarr = np.expand_dims(np.asarray(imgarr), axis=0)
        print('dims:',imgarr.shape)
        prediction = model.predict(imgarr, batch_size=1)
        prediction[prediction>0.5]=1
        prediction[prediction<0.5]=0
        prediction = np.squeeze(prediction)
        prediction=prediction.astype(np.int16)
        prediction=np.swapaxes(prediction,0,2)
        saveimg=sitk.GetImageFromArray(prediction)
        saveimg.SetSpacing(img.GetSpacing())
        saveimg.SetOrigin(img.GetOrigin())
        saveimg.SetDirection(img.GetDirection())
        #prediction在现在shape下的dice
        pred_dice = test_dice_coef(maskarr,prediction)
        dice_file.write(name+'---')
        dice_file.write(str(pred_dice))
        dice_file.write('\n')
        sitk.WriteImage(saveimg,os.path.join(r'H:\mutil_CT_seg\result_srm\unet3d\Prediction',name))
    
    dice_file.close()

main()