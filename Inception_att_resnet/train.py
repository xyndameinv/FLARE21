# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:16:55 2019

@author: Tan
这里训练是在划分验证和训练块,单独对训练数据增广,与generat_whole对应使用,
在整图上进行测试.
"""
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from random import sample
import nibabel as nib
import numpy as np
import shutil
from random import shuffle
#from res_unet_from_T import segmentation_model
from inception_att_net import unet_model_3d
from generators import get_training_and_validation_generators
from training import load_old_model,train_model
from aug3d_whole import augment

config = dict()
config["patch_shape"] = None
config["img_shape"] = (128,128,96)
config['input_size'] = tuple(list(config["patch_shape"])+[1]) if config["patch_shape"] else tuple(list(config["img_shape"])+[1])
config['nb_classes'] = 5
config['initial_filter_num'] = 16
config['l2_regular'] = 0.0##注意同时修改model中偏函数的weight_decay参数
config['bn'] = False
config["instance_norm"] = True
config['deconv'] = False
config['drop'] = 0.3
config['depth'] = 5
config['activation_fun'] = 'sigmoid'
if config['nb_classes']>1:
    config['activation_fun'] = 'softmax'
config["batch_size"] = 1 #在多个GPU下，保证val_num/batch_size>gpu_num
config["n_epochs"] = 50
config["initial_lr"] = 0.001 #与模型选择的optimizer对应，该处使用SGD

config["early_stop"] = 10 ## training will be stopped after this many epochs without the validation loss improving
config["learning_rate_drop"] =0.5
config["learning_rate_epochs"] = 5#None
config['learning_rate_patience'] = None

#训练数据的随机切块参数
config["binary_num_rate"] = 0.015 if config["patch_shape"] else None
config["min_point"] = 200 if config["patch_shape"] else None
config["pos_num_rate"] = 1.5 if config["patch_shape"] else None
config["nag_num_rate"] = 0.5 if config["patch_shape"] else None
config["validation_patch_overlap"]=(48,48,8) if config["patch_shape"] else None
config["validation_split"] = 0.1#0.1 #如果是None,则不划分yanzheng2集,45个完全训练

config["aug_times"] = 4#进行3倍的数据扩充
config["flip"] = True  # augments the data by randomly flipping an axis during
config["scale"] = True  # scale augment
config["rotate"] = True
config["augment"] = config["flip"] or config["scale"] or config["rotate"]

config["save_fpath"] = r'..\result_x1\incep_att_resnet15'
if not os.path.exists(config["save_fpath"]):
    os.mkdir(config['save_fpath'])
config["data_fpath"] = r'..\data\no_aug'
config["model_file"] = None#载入预训练模型
config["training_config_file"] = config['save_fpath']+'/confing.txt'
config_file=open(config['training_config_file'],'w')
for key,value in config.items():
    config_file.writelines(key+':'+str(value)+'\n')
config_file.close()

def get_data(data_fpath, split_rate):
    nii_dirs = sorted(os.listdir(os.path.join(data_fpath,'TrainImg')),key=str.lower)
    print(type(nii_dirs))
    #nii_dirs=nii_dirs[0:150]
    #nii_dirs=shuffle(nii_dirs)
    print(nii_dirs)
    # 获取训练验证的数据索引
    train_name = [];val_data=[];val_name = []
    if split_rate:
        nums = int(len(nii_dirs) * split_rate)
    else:
        nums=0
    #val_list = sample(nii_dirs,nums)
    val_list = nii_dirs[:nums]

    for i in nii_dirs:
        name = i
        if i in val_list:
            val_name.append(name)
            #files = sorted(glob.glob(os.path.join(data_fpath, i, '*', '*')),key=str.lower)
            img = nib.load(os.path.join(data_fpath,'TrainImg', i))
            image = np.expand_dims(img.get_fdata(), axis=-1).astype(np.float32)
            affine = img.affine;
            hdr=img.header
    
            truth = nib.load(os.path.join(data_fpath,'TrainMask', i))
            truth = np.expand_dims(truth.get_fdata(), axis=-1).astype(np.uint8)
            val_data.append(tuple([name, image, truth, affine,hdr]))
            print('val-- ',i,truth.max())
        else:
            train_name.append(name)
    return val_data,train_name, val_name

def main():
    print('loading training data.....')
    val_data,train_name, val_name = get_data(config['data_fpath'],split_rate=config["validation_split"])
    config_file = open(config['training_config_file'], 'a')
    config_file.write('train_nii: ' + ' / '.join(train_name))
    config_file.write('\n')
    config_file.write('val_nii: ' + ' / '.join(val_name))
    
    config_file.write('\n')
    config_file.write('augmentation info:\n')
    train_data = []
    for name in train_name:
        #files = glob.glob(os.path.join(config['data_fpath'], name, '*', '*'))
        #print(files)
        itk_img = nib.load(os.path.join(config['data_fpath'], 'TrainImg',name))
        itk_mask=nib.load(os.path.join(config['data_fpath'], 'TrainMask',name))
        image = np.expand_dims(itk_img.get_fdata(), axis=-1).astype(np.float32)
        print('after dims: ', image.shape)
        truth = np.expand_dims(itk_mask.get_fdata(), axis=-1).astype(np.uint8)
        train_data.append(tuple([name, image, truth, itk_img.affine,itk_img.header]))
        print('\ntrain-- ',name,truth.max())
        
        if config["augment"]:
            if not os.path.exists(os.path.join(r'..\data', 'aug')):
                os.mkdir(os.path.join(r'..\data', 'aug'))
            augment_times = config["aug_times"]
            aug_train = os.path.join(r'..\data', 'aug', 'TrainImg')
            if not os.path.exists(aug_train):
                os.mkdir(aug_train)
            aug_mask = os.path.join(r'..\data', 'aug', 'TrainMask')
            if not os.path.exists(aug_mask):
                os.mkdir(aug_mask)
            shutil.copy(os.path.join(config['data_fpath'], 'TrainImg',name),os.path.join(aug_train,name));shutil.copy(os.path.join(config['data_fpath'], 'TrainMask',name),os.path.join(aug_mask,name))
            while augment_times:
                augment_times-=1
                out_img,out_mask,info = augment(itk_img,itk_mask,flip=config["flip"],flip_axis=[0,1,2],
                                                    scale=config["scale"],factor_range=[0.8,0.9,1.1,1.2,],
                                                    rotate=config["rotate"],theta_range=[90,180,270])
                #shutil.copy(os.path.join(config['data_fpath'], 'TrainImg',name),aug_train);shutil.copy(os.path.join(config['data_fpath'], 'TrainMask',name),aug_mask)
                #aug_path = os.path.join(aug_folders, str(config["aug_times"] - augment_times))
                #os.mkdir(aug_path)
                savename=list(name)
                savename.insert(-7,'_'+str(augment_times))
                nib.save(out_img,os.path.join(aug_train, ''.join(savename)))
                nib.save(out_mask,os.path.join(aug_mask,  ''.join(savename)))
                aug_name = ''.join([name, '_', str(config["aug_times"] - augment_times)])
                config_file.write('{}: {}\n'.format(aug_name, info))
                out_image = np.expand_dims(out_img.get_fdata(), axis=-1).astype(np.float32)
                out_truth = np.expand_dims(out_mask.get_fdata(), axis=-1).astype(np.uint8)
                train_data.append(tuple([aug_name, out_image, out_truth, out_img.affine, out_img.header]))
    config_file.close()

    if config["model_file"]:
        print('load pre_train_model....')
        model = load_old_model(config["model_file"])
    else:
        print('build the new mode.....')
        model = unet_model_3d(input_shape=config["input_size"], n_base_filters=config['initial_filter_num'],n_labels=config["nb_classes"],
                                   initial_learning_rate=config["initial_lr"],activation_name=config['activation_fun'])

    # get training and testing generators
    print('get training and testing generators....')
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
            train_data,val_data,patch_shape=config["patch_shape"],batch_size=config["batch_size"],n_labels=config['nb_classes'],
            binary_num_rate=config["binary_num_rate"],min_point=config["min_point"],pos_num_rate=config["pos_num_rate"],
            nag_num_rate=config["nag_num_rate"],validation_patch_overlap=config["validation_patch_overlap"])

    # run training
    print('training is going on...')
    model = train_model(model=model,save_path=config['save_fpath'],
                initial_lr=config["initial_lr"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_epochs=config['learning_rate_epochs'],
                learning_rate_patience=config['learning_rate_patience'],
                n_epochs=config["n_epochs"],
                early_stopping_patience=config["early_stop"])
if __name__ == "__main__":
    main()
    print('finisn training....')