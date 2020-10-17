from tensorflow import keras
import tensorflow as tf
if not tf.__version__.startswith('1'):
    import tensorflow.compat.v1 as tf
import numpy as np
import logging


def convert_upsample(node, params, layers, lambda_func, node_name, keras_name):
    """
    Convert upsample.
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param lambda_func: function for keras Lambda layer
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    logger = logging.getLogger('onnx2keras:upsample')
    logger.warning('!!! EXPERIMENTAL SUPPORT (upsample) !!!')

    if len(node.input) > 2:
        raise AttributeError('Unsupported number of inputs')

    if params['mode'].decode('utf-8') == 'linear':
        sess = tf.InteractiveSession()
        scales = layers[node.input[1]]
        scale = (int(scales[2]), int(scales[3]))
        sess.close()

        upsampling = keras.layers.UpSampling2D(
            size=scale,
            name=keras_name,
            # data_format='channels_first',
            interpolation="bilinear"
        )

        layers[node_name] = upsampling(layers[node.input[0]])
    elif params['mode'].decode('utf-8') == 'nearest':
        sess = tf.InteractiveSession()
        scales = layers[node.input[1]]
        scale = (int(scales[2]), int(scales[3]))
        sess.close()

        upsampling = keras.layers.UpSampling2D(
            size=scale,
            name=keras_name,
            # data_format='channels_first',
            interpolation='nearest'
        )

        layers[node_name] = upsampling(layers[node.input[0]])
    else:
        logger.error('Cannot convert non-linear/non-nearest upsampling.')
        raise AssertionError('Cannot convert non-linear/non-nearest upsampling')
