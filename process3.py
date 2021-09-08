import os
import SimpleITK as sitk
import numpy as np
from shutil import copyfile
names=os.listdir(r'.\data\no_aug\TrainMask')
for i in range(300):
    name=names[i]
    print(name)
    copyfile(os.path.join(r'.\data\no_aug\TrainMask',name),os.path.join(r'.\data\SRM\TrainImg',name))
    img=sitk.ReadImage(os.path.join(r'.\data\no_aug\TrainMask',name))
    imgarr=sitk.GetArrayFromImage(img)
    imgarr[imgarr==3]=1
    imgarr[imgarr!=1]=0
    saveimg=sitk.GetImageFromArray(imgarr)
    saveimg.SetOrigin(img.GetOrigin())
    saveimg.SetDirection(img.GetDirection())
    saveimg.SetSpacing(img.GetSpacing())
    sitk.WriteImage(saveimg,os.path.join(r'.\data\SRM\TrainMask',name))
    