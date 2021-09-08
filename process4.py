import SimpleITK as sitk
import numpy as np
import os
import cv2

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
def getRangImageDepth(image):
    """
    args:
    image ndarray of shape (depth, height, weight)
    """
    # 得到轴向上出现过目标（label>=1)的切片
    ### image.shape为(depth,height,width),否则要改axis的值
    z = np.any(image, axis=(1, 2)) # z.shape:(depth,)
    ###  np.where(z)[0] 输出满足条件，即true的元素位置，返回为一个列表，用0，-1取出第一个位置和最后一个位置的值
    startposition, endposition = np.where(z)[0][[0, -1]]
    return startposition, endposition

def caijian(image, z):
    ls = []
    start = 0
    end = z
    for i in range(z):
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
    return start, end

    
path = r'.\data\Raw\TrainingImg'
m_path = r'.\data\Raw\TrainingMask'
names = os.listdir(path)
clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
for name in names:
    print(name)
    img = sitk.ReadImage(os.path.join(path, name))
    name = name.replace('_0000', '')
    mask = sitk.ReadImage(os.path.join(m_path, name))
    #图像调整窗位窗宽
    ct_array = sitk.GetArrayFromImage(img)
    ct_adjust = window_transform(ct_array, 400, 40, normal=False)
    saveimg=sitk.GetImageFromArray(ct_adjust)
    saveimg.SetOrigin(img.GetOrigin())
    saveimg.SetDirection(img.GetDirection())
    saveimg.SetSpacing(img.GetSpacing())
    
    #标签处理
    maskarr=sitk.GetArrayFromImage(mask)
    maskarr[maskarr>0]=1
    savemask=sitk.GetImageFromArray(maskarr)
    savemask.SetOrigin(mask.GetOrigin())
    savemask.SetDirection(mask.GetDirection())
    savemask.SetSpacing(mask.GetSpacing())
    
    #图像与标签resize 128 128 96
    resize_img=resize_image_itk(saveimg, (128,128,96),resamplemethod= sitk.sitkLinear)
    resize_mask=resize_image_itk(savemask, (128,128,96),resamplemethod= sitk.sitkNearestNeighbor)

    #图像标准化
    resize_imgarr=sitk.GetArrayFromImage(resize_img)#96 128 128
    nor_resize_imgarr=normalize_data(resize_imgarr)
    nor_resize_img=sitk.GetImageFromArray(nor_resize_imgarr)
    nor_resize_img.SetSpacing(resize_img.GetSpacing())
    nor_resize_img.SetOrigin(resize_img.GetOrigin())
    nor_resize_img.SetDirection(resize_img.GetDirection())

    #图像与标签保存
    sitk.WriteImage(nor_resize_img, os.path.join(r'.\data\no_aug_1\TrainImg',name))
    sitk.WriteImage(resize_mask, os.path.join(r'.\data\no_aug_1\TrainMask',name))
    