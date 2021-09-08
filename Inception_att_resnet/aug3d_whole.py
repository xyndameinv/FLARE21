#-*-coding:utf-8-*-
import scipy.ndimage as sni
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import random
from nilearn.image import new_img_like
import os
import shutil
    
def scale_image(affine,img_shape,scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(affine)
    new_affine[:3, :3] = affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = affine[:, 3][:3] + (img_shape * np.diag(affine)[:3] * (1 - scale_factor)) / 2
    return new_affine

def rotateit(image, mask,theta):
    #旋转
    #order = 0 if isseg == True else 5
    newimg = sni.rotate(image,theta,reshape=False, order=5, mode='nearest')
    newmask = sni.rotate(mask,theta,reshape=False, order=0, mode='nearest')
    return newimg,newmask

def flipit(image,mask, axes):
    #翻转
    if axes==0:
        newimg = np.fliplr(image)
        newmask = np.fliplr(mask)
    elif axes==1:
        newimg = np.flipud(image)
        newmask = np.flipud(mask)
    else:
        newimg = np.flip(image, 2)
        newmask = np.flip(mask, 2)
        #newimg = np.flip(image,0)
        #newmask = np.flip(mask,0)
        #newimg = np.flip(newimg,1)
        #newmask = np.flip(newmask,1)
    return newimg,newmask

def augment(itk_img,itk_mask,flip=True,flip_axis=[0,1,2],
            scale=True,factor_range=[0.8,0.9,1.0,1.1,1.2],
            rotate=True,theta_range=[90,180,270]):
    image = itk_img.get_fdata();hdr1=itk_img.header;hdr1.set_data_dtype(np.float32)
    mask = itk_mask.get_fdata();hdr2=itk_mask.header;hdr2.set_data_dtype(np.uint8)
    if flip:
        axis = random.sample(flip_axis,1)[0]
        image,mask = flipit(image,mask, axis)
    if rotate:
        theta = random.sample(theta_range,1)[0]
        image,mask = rotateit(image, mask,theta)
    if scale:
        scale_factor = random.sample(factor_range,1)[0] #随机产生旋转因子
        img_affine = itk_img.affine
        mask_affine = itk_mask.affine
        new_img_affine = scale_image(img_affine,image.shape,scale_factor)
        new_mask_affine = scale_image(mask_affine,mask.shape,scale_factor)
    else:
        new_img_affine = itk_img.affine
        new_mask_affine = itk_mask.affine
    info = 'flip_axie:'+str(axis)+'   '+'scale_factor:'+str(scale_factor)+'   '+'rotate_theta:'+str(theta)
    out_img = nib.Nifti1Image(image,new_img_affine,hdr1)
    out_mask = nib.Nifti1Image(mask, new_mask_affine, hdr2)
    return out_img,out_mask,info
