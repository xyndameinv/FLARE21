# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 20:17:00 2021

@author: Administrator
"""
import numpy as np
import SimpleITK as sitk
import os
names=os.listdir(r'H:\mutil_CT_seg\result_x1\incep_att_resnet9\ValPrediction_ori')
#os.mkdir(r'H:\mutil_CT_seg\result_x1\incep_att_resnet9\ValPrediction_ori_post')
for name in names:
    print(name)
    mask=sitk.ReadImage(os.path.join(r'H:\mutil_CT_seg\result_x1\incep_att_resnet9\ValPrediction_ori',name))
    maskarr=sitk.GetArrayFromImage(mask)
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    lastout=np.zeros(maskarr.shape)
    for i in np.unique(maskarr).tolist()[1:]:
        maskarr1=maskarr.copy()
        maskarr1[maskarr1!=i]=0
        #print(maskarr1.max())
        mask_c=sitk.GetImageFromArray(maskarr1)
        output=cca.Execute(mask_c)
        outputarr=sitk.GetArrayFromImage(output) #连通域labels
        #print(outputarr.max())
        if i==1:
            ##肝只保留最大的连通域(除背景)
            numlist=[]
            arealist=[]
            for j in range(outputarr.max()+1):
                numlist.append(j)
                arealist.append((outputarr==j).sum())
            num_list_sorted = sorted(numlist, key=lambda k: arealist[k], reverse=True)
            #print(num_list_sorted)
            outputarr[outputarr!=num_list_sorted[1]]=0
            outputarr[outputarr==num_list_sorted[1]]=i
            #print(outputarr.max())
            lastout=lastout+outputarr
        elif i==2:
            #肾保留最大的两个连通域(看看前两个连通域的大小 有可能出现)
            numlist=[]
            arealist=[]
            for j in range(outputarr.max()+1):
                numlist.append(j)
                arealist.append((outputarr==j).sum())
            num_list_sorted = sorted(numlist, key=lambda k: arealist[k], reverse=True)
            #print(num_list_sorted)
            if len(num_list_sorted)>2 and arealist[num_list_sorted[2]]>100:
                noselect=np.logical_and(outputarr!=num_list_sorted[1],outputarr!=num_list_sorted[2])
                select=np.logical_or(outputarr==num_list_sorted[1],outputarr==num_list_sorted[2])
                outputarr[noselect]=0
                outputarr[select]=i
            else:
                outputarr[outputarr!=num_list_sorted[1]]=0
                outputarr[outputarr==num_list_sorted[1]]=i
            #print(outputarr.max())
            lastout=lastout+outputarr
        elif i==3:
            #脾保留最大的一个连通域
            numlist=[]
            arealist=[]
            for j in range(outputarr.max()+1):
                numlist.append(j)
                arealist.append((outputarr==j).sum())
            num_list_sorted = sorted(numlist, key=lambda k: arealist[k], reverse=True)
            #print(num_list_sorted)
            outputarr[outputarr!=num_list_sorted[1]]=0
            outputarr[outputarr==num_list_sorted[1]]=i
            #print(outputarr.max())
            lastout=lastout+outputarr
        else:
            #胰腺保留最大的两个连通域
            numlist=[]
            arealist=[]
            for j in range(outputarr.max()+1):
                numlist.append(j)
                arealist.append((outputarr==j).sum())
            num_list_sorted = sorted(numlist, key=lambda k: arealist[k], reverse=True)
            #print(num_list_sorted)
            if len(num_list_sorted)>2 and arealist[num_list_sorted[2]]>100:
                noselect=np.logical_and(outputarr!=num_list_sorted[1],outputarr!=num_list_sorted[2])
                select=np.logical_or(outputarr==num_list_sorted[1],outputarr==num_list_sorted[2])
                outputarr[noselect]=0
                outputarr[select]=i
            else:
                outputarr[outputarr!=num_list_sorted[1]]=0
                outputarr[outputarr==num_list_sorted[1]]=i
            #print(outputarr.max())
            lastout=lastout+outputarr
    lastout=lastout.astype('uint8')
    outmask=sitk.GetImageFromArray(lastout)
    outmask.SetOrigin(mask.GetOrigin())
    outmask.SetSpacing(mask.GetSpacing())
    outmask.SetDirection(mask.GetDirection())
    sitk.WriteImage(outmask,os.path.join(r'H:\mutil_CT_seg\result_x1\incep_att_resnet9\ValPrediction_ori_post',name))