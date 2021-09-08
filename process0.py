# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:46:55 2021

@author: Administrator
"""
import SimpleITK as sitk
import numpy as np
import os
import cv2
from shutil import copyfile
import random

num=379
lists=['train_002_0000.nii.gz','train_019_0000.nii.gz','train_069_0000.nii.gz','train_101_0000.nii.gz','train_114_0000.nii.gz','train_127_0000.nii.gz','train_150_0000.nii.gz','train_134_0000.nii.gz','train_174_0000.nii.gz','train_195_0000.nii.gz']
for name in lists:
    num=num+1
    copyfile(os.path.join(r'.\data\Raw\TrainingImg',name),os.path.join(r'.\data\Raw\TrainingImg','train_'+str(num)+'_0000.nii.gz'))
    copyfile(os.path.join(r'.\data\Raw\TrainingMask',name.replace('_0000','')),os.path.join(r'.\data\Raw\TrainingMask','train_'+str(num)+'.nii.gz'))
    num=num+1
    img=sitk.ReadImage(os.path.join(r'.\data\Raw\TrainingImg',name))
    mask=sitk.ReadImage(os.path.join(r'.\data\Raw\TrainingMask',name.replace('_0000','')))
    imgarr=sitk.GetArrayFromImage(img)
    maskarr=sitk.GetArrayFromImage(mask)
    imgarr1=imgarr.copy()
    imgarr1[maskarr!=2]=0
    imgarr2=imgarr1.copy()
    imgarr2[imgarr2>50]=0
    for i in range(imgarr2.shape[0]):
        for j in range(imgarr2.shape[1]):
            for k in range(imgarr2.shape[2]):
                if imgarr2[i,j,k]!=0:
                    imgarr2[i,j,k]=imgarr2[i,j,k]+random.randint(100,150)
    imgarr1[imgarr1<=50]=0
    imgarr1=imgarr1+imgarr2
    imgarr[maskarr==2]=0
    imgarr=imgarr+imgarr1
    saveimg=sitk.GetImageFromArray(imgarr)
    saveimg.SetSpacing(img.GetSpacing())
    saveimg.SetDirection(img.GetDirection())
    saveimg.SetOrigin(img.GetOrigin())
    sitk.WriteImage(saveimg,os.path.join(r'.\data\Raw\TrainingImg','train_'+str(num)+'_0000.nii.gz'))
    sitk.WriteImage(mask,os.path.join(r'.\data\Raw\TrainingMask','train_'+str(num)+'.nii.gz'))