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
import time
import cv2
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from metrics import (dice_coefficient, dice_coefficient_loss,weighted_dice_coefficient_loss1, weighted_dice_coefficient)

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
def window_transform(ct_array, windowWidth, windowCenter, normal=False):
   """
   return: trucated image according to window center and window width
   and normalized to [0,1]
   """
   minWindow = float(windowCenter) - 0.5*float(windowWidth)
   newimg = (ct_array - minWindow) / float(windowWidth)
   newimg[newimg < 0] = 0
   newimg[newimg > 1] = 1
   if not normal:
   	newimg = (newimg * 255).astype('uint8')
   return newimg
   
def post_processing(maskarr,reference):
    #select connect
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
    maskarr=lastout
    #fill
    BMC=sitk.BinaryMorphologicalClosingImageFilter()
    BMC.SetKernelType(sitk.sitkBall)
    BMC.SetKernelRadius(2)
    BMC.SetForegroundValue(3)
    mask_c=sitk.GetImageFromArray(maskarr)
    out=BMC.Execute(mask_c)
    outarr=sitk.GetArrayFromImage(out)
    return outarr
    
def post_processing1(maskarr):
    #select connect
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    mask_c=sitk.GetImageFromArray(maskarr)
    output=cca.Execute(mask_c)
    outputarr=sitk.GetArrayFromImage(output)
    arealist=[]
    numlist=[]
    for j in range(outputarr.max()+1):
        numlist.append(j)
        arealist.append((outputarr==j).sum())
        if (outputarr==j).sum()<1500:
            outputarr[outputarr==j]=0
    #num_list_sorted = sorted(numlist, key=lambda k: arealist[k], reverse=True)
    #outputarr[outputarr==num_list_sorted[0]]=0
    outputarr[outputarr!=0]=1
    return outputarr
    
def caijian(image, z):
    ls = []
    start = 0
    end = 95
    for i in range(96):
        exist = (image[i, :, :] > 0) * 1
        # factor = np.ones(x, y)
        # res = np.dot(exist, factor)
        a = np.sum(exist)
        if a < 50:
            ls.append(0)
        else:
            ls.append(a)
    for i in range(len(ls)):
        if ls[i] != 0:
            start = i
            break
    for j in range(len(ls)-1, 0, -1):
        if ls[j] != 0:
            end = j
            break
    # print(start, end)

    img_start = round(start / 96 * z)
    img_end = round(z - (96 - end)/96 * z)
    if img_start - 10 < 0:
        img_start = 0
    else:
        img_start = img_start - 10
    if img_end + 10 >= z:
        img_end = z - 1
    else:
        img_end = img_end + 10
    return img_start, img_end
    
def load_old_model1(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss,'dice_coefficient': dice_coefficient,
                      'clas_0_dice': dice_coefficient,'clas_1_dice': dice_coefficient,'clas_2_dice': dice_coefficient,
                      'clas_3_dice': dice_coefficient,'clas_4_dice': dice_coefficient,'clas_5_dice': dice_coefficient,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss1,
                      'InstanceNormalization': InstanceNormalization}
    return load_model(model_file, custom_objects=custom_objects)
    
def main():
    names=os.listdir(r'..\data\Raw\ValidationImg')
    model=load_old_model(r'.\model.20-0.02.hdf5')
    binary_model=load_old_model1(r'.\crop_model.hdf5')
    start=time.time()
    for name in names:
        print(name)
        img=sitk.ReadImage(os.path.join(r'..\data\Raw\ValidationImg',name))
        origin_size=img.GetSize()
        '''crop'''
        #adjust window and resize 
        ct_array = sitk.GetArrayFromImage(img)
        ct_adjust = window_transform(ct_array, 400, 40, normal=False)
        saveimg=sitk.GetImageFromArray(ct_adjust)
        saveimg.SetOrigin(img.GetOrigin())
        saveimg.SetDirection(img.GetDirection())
        saveimg.SetSpacing(img.GetSpacing())
        #resize and normalize
        resize_img=resize_image_itk(saveimg, (128,128,96),resamplemethod= sitk.sitkLinear)
        resize_imgarr=sitk.GetArrayFromImage(resize_img)#96 128 128
        swap_resize_imgarr=np.swapaxes(resize_imgarr,0,2) #128 128 96
        swap_resize_imgarr=normalize_data(swap_resize_imgarr)
        #将swap_resize_imgarr输入网络测试 得到prediction
        imgarr = np.expand_dims(np.asarray(swap_resize_imgarr), axis=-1)
        imgarr = np.expand_dims(np.asarray(imgarr), axis=0)
        prediction = binary_model.predict(imgarr, batch_size=1)
        prediction[prediction>0.5]=1
        prediction[prediction<0.5]=0
        prediction = np.squeeze(prediction)
        prediction=prediction.astype(np.int16)
        prediction=np.swapaxes(prediction,0,2) #96 128 128
        #post processing
        prediction=post_processing1(prediction)
        #caijian 
        start,end=caijian(prediction,origin_size[2])
        crop_img=ct_adjust[start:end,:,:]
        #save current image
        saveimg=sitk.GetImageFromArray(crop_img)
        saveimg.SetDirection(img.GetDirection())
        saveimg.SetSpacing(img.GetSpacing())
        cropshape=saveimg.GetSize()
        '''crop over'''
        #resize and normalize
        resize_img=resize_image_itk(saveimg, (128,128,96),resamplemethod= sitk.sitkLinear)
        resize_imgarr=sitk.GetArrayFromImage(resize_img)#96 128 128
        swap_resize_imgarr=np.swapaxes(resize_imgarr,0,2) #128 128 96
        swap_resize_imgarr=normalize_data(swap_resize_imgarr)
        #将swap_resize_imgarr输入网络测试 得到prediction
        imgarr = np.expand_dims(np.asarray(swap_resize_imgarr), axis=-1)
        imgarr = np.expand_dims(np.asarray(imgarr), axis=0)
        prediction = model.predict(imgarr, batch_size=1)
        prediction = np.squeeze(np.argmax(prediction,axis=-1),axis=0)
        prediction=prediction.astype(np.int16)
        prediction=np.swapaxes(prediction,0,2) #96 128 128
        #post processing
        prediction=post_processing(prediction,img)
        #prediction恢复原大小并保存
        saveprediction=sitk.GetImageFromArray(prediction)
        saveprediction.SetSpacing(resize_img.GetSpacing())
        saveprediction.SetOrigin(resize_img.GetOrigin())
        saveprediction.SetDirection(resize_img.GetDirection())
        resize_prediction=resize_image_itk(saveprediction, cropshape,resamplemethod= sitk.sitkNearestNeighbor)
        resize_prediction_arr=sitk.GetArrayFromImage(resize_prediction)
        
        orginmask=np.zeros((origin_size[2],origin_size[1],origin_size[0]))
        orginmask[start:end]=resize_prediction_arr
        lastmask=sitk.GetImageFromArray(orginmask)
        lastmask.SetSpacing(img.GetSpacing())
        lastmask.SetOrigin(img.GetOrigin())
        lastmask.SetDirection(img.GetDirection())
        #保存预测结果
        if not os.path.exists(r'H:\mutil_CT_seg\result_x1\incep_att_resnet13\ValPrediction_ori2'):
            os.mkdir(r'H:\mutil_CT_seg\result_x1\incep_att_resnet13\ValPrediction_ori2')
        sitk.WriteImage(lastmask,os.path.join(r'H:\mutil_CT_seg\result_x1\incep_att_resnet13\ValPrediction_ori2',name))
    end=time.time()
    alltime=end-start
    print(alltime)
if __name__ == "__main__":
    main()