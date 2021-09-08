##our method
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation,Reshape,PReLU, Deconvolution3D,Add,SpatialDropout3D,\
                            add,GlobalAveragePooling3D,AveragePooling3D,multiply,Lambda,Dense
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
#from keras.utils import plot_model
#from keras.utils import multi_gpu_model

from metrics import (dice_coefficient_loss, get_label_dice_coefficient_function,dice_coefficient,
                     weighted_dice_coefficient_loss,weighted_dice_coefficient)

K.set_image_data_format("channels_last")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate



def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def expand_dim_backend(x):
    x = K.expand_dims(x,1)
    x = K.expand_dims(x,1)
    x = K.expand_dims(x,1)
    return x
def senet(layer, n_filter):
    seout = GlobalAveragePooling3D()(layer)
    seout = Dense(units=int(n_filter/2))(seout)
    seout = Activation("relu")(seout)
    seout = Dense(units=n_filter)(seout)
    seout = Activation("sigmoid")(seout)
    print("seout1 shape",seout.shape)
    # seout = Reshape([-1,1,1,n_filter])(seout)
    seout = Lambda(expand_dim_backend)(seout)
    print("seout shape",seout.shape)
    return seout

def resudial_block(input_layer, n_filters,kernel_1=(1, 1, 1),kernel_3=(3,3,3), padding='same', strides=(1, 1, 1)):
    layer = input_layer

    layer_1 = Conv3D(int(n_filters/2), kernel_1, padding=padding, strides=strides)(layer)
    layer_1 = InstanceNormalization(axis=-1)(layer_1)
    layer_1 = Activation("relu")(layer_1)

    layer_2 = Conv3D(int(n_filters/4),kernel_1, padding=padding, strides=strides)(layer)
    layer_2 = InstanceNormalization(axis=-1)(layer_2)
    layer_2 = Activation('relu')(layer_2)
    layer_2 = Conv3D(int(n_filters/2),kernel_3, padding=padding, strides=strides)(layer_2)
    layer_2 = InstanceNormalization(axis=-1)(layer_2)
    layer_2 = Activation('relu')(layer_2)

    layer_3 = Conv3D(int(n_filters/4),kernel_1, padding=padding, strides=strides)(layer)
    layer_3 = InstanceNormalization(axis=-1)(layer_3)
    layer_3 = Activation('relu')(layer_3)
    layer_3 = Conv3D(int(n_filters*3/4),kernel_3, padding=padding, strides=strides)(layer_3)
    layer_3 = InstanceNormalization(axis=-1)(layer_3)
    layer_3 = Activation('relu')(layer_3)
    layer_3 = Conv3D(int(n_filters),kernel_3, padding=padding, strides=strides)(layer_3)
    layer_3 = InstanceNormalization(axis=-1)(layer_3)
    layer_3 = Activation('relu')(layer_3)
    

    layer = concatenate([layer_1,layer_2,layer_3],axis=-1)

    seout = senet(layer,int(n_filters*2))  #注意力模块
    seout = multiply([seout,layer])  #注意力模块得到的结果与之前的乘 表示的是各个通道的重要程度不同

    layer = Conv3D(n_filters, kernel_1, padding=padding, strides=strides)(seout)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = Activation("relu")(layer)

    x_short = Conv3D(n_filters,kernel_1,padding=padding,strides=strides)(input_layer)

    layer_out = add([x_short, layer])

    return layer_out

def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=5e-3, deconvolution=False,
                  depth=4, n_base_filters=32, loss_function=weighted_dice_coefficient_loss,
                      metrics=[weighted_dice_coefficient],activation_name="sigmoid",include_label_wise_dice_coefficients=True,optimizer=Adam):
    ############################
    #resUnet + dropout
    ###############################
    inputs = Input(input_shape)
    # current_layer = inputs    
    inputs_1 = Conv3D(n_base_filters,kernel_size=(3,3,3),strides=(1,1,1),padding="same")(inputs)
    inputs_1 = InstanceNormalization(axis=-1)(inputs_1)
    inputs_1 = Activation("relu")(inputs_1)
    layer1 = resudial_block(inputs_1,n_base_filters)
    layer1_pool = MaxPooling3D(pool_size=(2,2,2))(layer1)

    layer2 = resudial_block(layer1_pool,n_base_filters*2)
    layer2_poo2 = MaxPooling3D(pool_size=pool_size)(layer2)

    layer3 = resudial_block(layer2_poo2,n_base_filters*4)
    layer3_poo3 = MaxPooling3D(pool_size=pool_size)(layer3)
    
    layer4 = resudial_block(layer3_poo3,n_base_filters*8)
    layer4_poo4 = MaxPooling3D(pool_size=pool_size)(layer4)

    layer4_poo4 = SpatialDropout3D(rate=0.1)(layer4_poo4)
    layer5 = Conv3D(n_base_filters*16, kernel_size=(3,3,3), padding="same", strides=(1,1,1))(layer4_poo4)
    layer5 = InstanceNormalization(axis=-1)(layer5)
    layer5 = Activation("relu")(layer5)
    layer5 = Conv3D(n_base_filters * 16, kernel_size=(3, 3, 3), padding="same", strides=(1, 1, 1))(layer5)
    layer5 = InstanceNormalization(axis=-1)(layer5)
    layer5 = Activation("relu")(layer5)
    layer5 = SpatialDropout3D(rate=0.1)(layer5)
    
    layer_up_4 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                            n_filters=n_base_filters*3)(layer5)
    concat4 = concatenate([layer_up_4, layer4], axis=-1)
    layer44 = resudial_block(concat4,n_base_filters*8)

    layer_up_3 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                            n_filters=n_base_filters*3)(layer44)
    concat3 = concatenate([layer_up_3, layer3], axis=-1)
    layer33 = resudial_block(concat3,n_base_filters*4)

    layer_up_2 = get_up_convolution(pool_size=pool_size, deconvolution=False,
                                    n_filters=n_base_filters * 2)(layer33)
    concat2 = concatenate([layer_up_2, layer2], axis=-1)
    layer22 = resudial_block(concat2, n_base_filters * 2)

    layer_up_1 = get_up_convolution(pool_size=(2,2,2), deconvolution=False,
                                    n_filters=n_base_filters * 1)(layer22)
    concat1 = concatenate([layer_up_1, layer1], axis=-1)
    layer11 = resudial_block(concat1, n_base_filters * 1)
    
    final_convolution = Conv3D(n_labels, (1, 1, 1), padding="same", strides=(1, 1, 1))(layer11)
    print("final_convolution.shape:",final_convolution.shape)
    act = Activation(activation_name)(final_convolution)
    print("act.shape:", act.shape)
    model = Model(inputs=inputs, outputs=act)

    #plot_model(model, to_file='model.png')
    if not isinstance(metrics, list):
        metrics = [metrics]

    if n_labels > 1 and include_label_wise_dice_coefficients:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(0,n_labels)]
        metrics.extend(label_wise_dice_metrics)
        print('multi_class....')

    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function,metrics=metrics)

    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer.lr
    #     return lr
    # lr_metric = get_lr_metric(Adam(lr=initial_learning_rate))
    print(model.summary())
    return model