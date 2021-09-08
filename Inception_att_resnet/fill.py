import SimpleITK as sitk
import numpy as np
import os 

names=os.listdir(r'H:\mutil_CT_seg\result_x1\incep_att_resnet9\ValPrediction_ori_post')
BMC=sitk.BinaryMorphologicalClosingImageFilter()
BMC.SetKernelType(sitk.sitkBall)
BMC.SetKernelRadius(10)
BMC.SetForegroundValue(1)
for name in names:
    print(name)
    mask=sitk.ReadImage(os.path.join(r'H:\mutil_CT_seg\result_x1\incep_att_resnet9\ValPrediction_ori_post',name))
    maskarr=sitk.GetArrayFromImage(mask)
    print('maskarr type:',maskarr.dtype)
    maskarr1=maskarr.copy()
    maskarr1[maskarr1!=3]=0
    maskarr1[maskarr1!=0]=1
    print('maskarr1 type:',maskarr1.dtype)
    mask_c=sitk.GetImageFromArray(maskarr1)
    mask_c.SetOrigin(mask.GetOrigin())
    mask_c.SetDirection(mask.GetDirection())
    mask_c.SetSpacing(mask.GetSpacing())
    out=BMC.Execute(mask_c)
    outarr=sitk.GetArrayFromImage(out)
    outarr[outarr==1]=3
    maskarr[maskarr==3]=0
    maskarr=maskarr+outarr
    saveimg=sitk.GetImageFromArray(maskarr)
    saveimg.SetOrigin(mask.GetOrigin())
    saveimg.SetDirection(mask.GetDirection())
    saveimg.SetSpacing(mask.GetSpacing())
    sitk.WriteImage(saveimg,os.path.join(r'H:\mutil_CT_seg\result_x1\incep_att_resnet9\ValPrediction_ori_post_fill',name))
    