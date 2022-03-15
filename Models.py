#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Mostafa Shabani (mshabani@ece.au.dk)
"""

import Layers as Layers
import keras
from keras import backend as K


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def TABL_2attention(template, dropout=0.1, projection_regularizer=None, projection_constraint=None,
                    attention_regularizer=None, attention_constraint=None, learn_rate=0.01, full_TABL=0,
                    L_concatenate=0):
    """
    Temporal Attention augmented Bilinear Layer network, refer to the paper https://arxiv.org/abs/1712.00975

    inputs
    ----

    template: a list of network dimensions including input and output, e.g., [[40,10], [120,5], [3,1]]
    dropout: dropout percentage
    projection_regularizer: keras regularizer object for projection matrices
    projection_constraint: keras constraint object for projection matrices
    attention_regularizer: keras regularizer object for attention matrices
    attention_constraint: keras constraint object for attention matrices
    L_concatenate : 1 concatenate on time
    L_concatenate : 2 concatenate on features
    full_TABL : zero: just the last is TABL and other layers are BL. one: all layers are TABL
    outputs
    ------
    keras model object
    """

    inputs = keras.layers.Input(template[0])

    x = inputs
    for k in range(1, len(template) - 1):
        if full_TABL == 0:
            x = Layers.BL(template[k], projection_regularizer, projection_constraint)(x)
        else:
            # L_concatenate= 1 concatenate on time
            # L_concatenate = 2 concatenate on features
            if L_concatenate == 0:
                x = Layers.TABL_2_attention(template[k], projection_regularizer, projection_constraint,
                                            attention_regularizer, attention_constraint)(x)
            elif L_concatenate == 1:
                x = Layers.TABL_2_attention_concatenate(template[k], projection_regularizer, projection_constraint,
                                                        attention_regularizer, attention_constraint)(x)
            else:
                x = Layers.TABL_2_attention_concatenate_on_features(template[k], projection_regularizer,
                                                                    projection_constraint,
                                                                    attention_regularizer, attention_constraint)(x)

        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)

    if L_concatenate == 0:
        x = Layers.TABL_2_attention(template[-1], projection_regularizer, projection_constraint,
                                    attention_regularizer, attention_constraint)(x)
    elif L_concatenate == 1:
        x = Layers.TABL_2_attention_concatenate(template[-1], projection_regularizer, projection_constraint,
                                                attention_regularizer, attention_constraint)(x)
    else:
        x = Layers.TABL_2_attention_concatenate_on_features(template[-1], projection_regularizer, projection_constraint,
                                                            attention_regularizer, attention_constraint)(x)

    outputs = keras.layers.Activation('softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(learn_rate, beta_1=0.9, beta_2=0.999)
    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer, 'categorical_crossentropy', ['acc', get_f1, lr_metric])

    return model
