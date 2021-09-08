# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:21:52 2019

@author: Tan
"""
from functools import partial
from keras import backend as K
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred, nb_class=2):
    if nb_class:
        loss = 0.0
        for label in range(0,nb_class):
            loss = loss+(1-dice_coefficient(y_true[:,:,:,:, label], y_pred[:,:,:,:, label]))
        return loss
    else:
        return 1-dice_coefficient(y_true, y_pred)

#附加模型，约束损失
#是否在载入其余模型用于辅助
'''additional_model = False
if additional_model:
    model_cae,encoder_model = d3_stacked_cae.bulid_net((64,64,32,1),0.001)
    #model_cae.load_weights('F:/2017/ty/prostate_3d_segmentation/result/cae_result/2019-3-10/ace_model.hdf5')
    #model_cae.trainable = False
    encoder_model.load_weights('E:/ty/prostate_3d_segmentation/result/cae_result/2019-3-10/encoder_model.hdf5')
    encoder_model.trainable = False
def bin_cros(y_true, y_pred):
    bin_crosstropy = K.mean(K.binary_crossentropy(y_true, y_pred)) #, axis=-1)
    return bin_crosstropy
def connection_loss(y_true, y_pred):
    seg_dice_loss = dice_coef_loss(y_true,y_pred)
#    constr_y_pred = model_cae(y_pred)
#    constr_dice_loss = dice_coef_loss(y_true,constr_y_pred)
    encoder_loss = bin_cros(encoder_model(y_true), encoder_model(y_pred))
#    return seg_dice_loss+0.001*constr_dice_loss+0.01*encoder_loss
    return seg_dice_loss+0.01*encoder_loss'''


def weighted_dice_coefficient(y_true, y_pred, axis=(1, 2, 3), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels last" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    '''
    dice = 0.0
    weights=[0.1,0.15,0.3,0.15,0.3]
    for label in range(0,5):
        dice = dice+dice_coefficient(y_true[:,:,:,:, label], y_pred[:,:,:,:, label])*weights[label]
    return dice
    '''
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))
def weighted_dice_coefficient_loss1(y_true, y_pred):
    return 1.0-weighted_dice_coefficient(y_true, y_pred)
    
def weighted_dice_coefficient_loss(y_true, y_pred):
    #return 1.0-weighted_dice_coefficient(y_true, y_pred)
    
    #dice loss
    loss1=1.0-weighted_dice_coefficient(y_true, y_pred)
    '''
    #重构loss
    prediction_p=srmmodel(y_pred)
    y_true_argmax=K.argmax(y_true,axis=-1) #0 1 2 3 4
    y_true_argmax=tf.where(tf.equal(y_true_argmax,2),0,y_true_argmax)
    y_true_argmax=tf.where(tf.equal(y_true_argmax,4),0,y_true_argmax)
    y_true_argmax=tf.where(tf.equal(y_true_argmax,3),1,y_true_argmax)
    loss2=1.0-weighted_dice_coefficient(prediction_p, y_true_argmax)
    '''
    #shape loss
    encoder_p=srmencoder(y_pred)
    encoder_t=srmencoder(y_true)
    loss3=K.mean(K.binary_crossentropy(encoder_p,encoder_t))
    #print('loss1:-----',loss1)
    #print('loss3:-----',loss3)
    return loss1+0.01*loss3
    
def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:,:,:,:, label_index], y_pred[:,:,:,:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'clas_{0}_dice'.format(label_index))
    return f

def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss,'dice_coefficient': dice_coefficient,
                      'clas_0_dice': dice_coefficient,'clas_1_dice': dice_coefficient,'clas_2_dice': dice_coefficient,
                      'clas_3_dice': dice_coefficient,'clas_4_dice': dice_coefficient,'clas_5_dice': dice_coefficient,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss1,
                      'InstanceNormalization': InstanceNormalization}
    return load_model(model_file, custom_objects=custom_objects)
    
dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
weighted_dice_coefficient=weighted_dice_coefficient
weighted_dice_coefficient_loss=weighted_dice_coefficient_loss
weighted_dice_coefficient_loss1=weighted_dice_coefficient_loss1
srmmodel=load_old_model(r'.\srm_model.hdf5')
srmencoder=Model(inputs=srmmodel.input,outputs=srmmodel.get_layer('activation_8').output)