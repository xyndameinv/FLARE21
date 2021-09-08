import os
import numpy as np
#os.environ["PATH"] += os.pathsep +'D:/Graphviz2.38/bin/'
import tensorflow as tf
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
runmeta = tf.compat.v1.RunMetadata()
import keras
#config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
 
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
from keras.utils import multi_gpu_model
from keras.engine import Input, Model
from keras.layers import (Conv3D,MaxPooling3D,UpSampling3D,concatenate,Activation,PReLU,
                          Deconvolution3D,SpatialDropout3D,BatchNormalization,LeakyReLU)
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from metrics import (dice_coefficient_loss, get_label_dice_coefficient_function,dice_coefficient,
                     weighted_dice_coefficient_loss,weighted_dice_coefficient)

from functools import partial
from keras import backend as K
K.set_image_data_format("channels_last")

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
    return tuple([None] + output_image_shape + [n_filters])

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False,weight_decay=0.0):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, kernel_regularizer=l2(weight_decay))(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=-1)(layer)
    elif instance_normalization:
        layer = InstanceNormalization(axis=-1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)



create_convolution_block = partial(create_convolution_block,weight_decay=0.0)
def unet_model_3d(input_shape, pool_size=(2, 2, 2), nb_classes=1, initial_learning_rate=0.00001, deconvolution=False,
                  drop = 0.3,weight_decay=1e-5,depth=4, n_base_filters=32, include_label_wise_dice_coefficients=TypeError,
                  metrics=weighted_dice_coefficient,loss=weighted_dice_coefficient_loss,batch_norm=True,
                  instance_norm=False,activation_name="sigmoid",multi_gpu=False):
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()
    
    #求解当前input_shape下允许的最大采样次数，主要针对z轴
    max_down_depth=0;d_shape=input_shape[-2]
    while d_shape%2==0:
        d_shape//=2
        max_down_depth+=1

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_norm,instance_normalization=instance_norm,
                                          weight_decay=weight_decay)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_norm,instance_normalization=instance_norm,
                                          weight_decay=weight_decay)
        if layer_depth < depth - 1:
            if layer_depth>=max_down_depth-3:
                current_layer = MaxPooling3D(pool_size=(2,2,2))(layer2)
            else:
                current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            if drop:
                current_layer = SpatialDropout3D(drop,data_format="channels_last")(layer2)
            else:
                current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        if layer_depth>=max_down_depth-3:
            up_convolution = get_up_convolution(pool_size=(2,2,2), deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[-1])(current_layer)
        else:
            up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[-1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=-1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                 input_layer=concat, batch_normalization=batch_norm,
                                                 instance_normalization=instance_norm,weight_decay=weight_decay)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[-1],
                                                 input_layer=current_layer,batch_normalization=batch_norm,
                                                 instance_normalization=instance_norm,weight_decay=weight_decay)

    final_convolution = Conv3D(nb_classes, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    print('act.shape:',act.shape)
    model = Model(inputs=inputs, outputs=act)
    
    if multi_gpu:
        model = multi_gpu_model(model, 2)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if nb_classes > 1 and include_label_wise_dice_coefficients:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(0,nb_classes)]
        metrics.extend(label_wise_dice_metrics)
        print('multi_class....')
        
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss, metrics=metrics)    
    print('use {}'.format(loss))
    #plot_model(model, to_file='d3_unet.png',show_shapes=True)
    print(model.summary())
    return model