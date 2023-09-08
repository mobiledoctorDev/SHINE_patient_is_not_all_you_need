# ----------------------------------------
# Standard modules
# ----------------------------------------
import sys

# --------------------------------------
# Deep Learning Modules
# --------------------------------------
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.initializers import Ones, Zeros
from keras.layers import Layer, Add, Dropout, Dense, Activation

# ----------------------------------------
# Common modules
# ----------------------------------------
sys.path.append("..")
from common import utils, configs

# ----------------------------------------
# Custom modules
# ----------------------------------------
from base_model import BaseModel

# --------------------------------------
# MLP class
# --------------------------------------
class MLP(BaseModel):
    def __init__(self, structure_params, ver, input_name, task, dict_encodings,
                 pretrained_model_name="random", is_visit_based=False, output_hxg=False, dict_output_classes=None):
        super(MLP, self).__init__(ver, input_name, task, dict_encodings,
                                  pretrained_model_name=pretrained_model_name, is_visit_based=is_visit_based,
                                  output_hxg=output_hxg, dict_output_classes=dict_output_classes)

        self.structure_params = structure_params

    def body(self, tensor):
        # initialize with class variables
        n_layers, n_units, use_dropout, use_bn = self.structure_params
        w_reg = self.w_reg

        x = tensor
        for layer_idx in range(n_layers):
            x = Dense(n_units, kernel_regularizer=w_reg, name="dense{}".format(layer_idx + 1))(x)
            if use_bn:
                x = BatchNormalization(axis=-1, gamma_regularizer=w_reg, beta_regularizer=w_reg,
                                       name="bn{}".format(layer_idx + 1))(x)
            x = Activation("relu", name="act{}".format(layer_idx + 1))(x)
            if use_dropout:
                x = Dropout(0.5, name="dropout{}".format(layer_idx + 1))(x)

        return x

# --------------------------------------
# ResMLP class
# --------------------------------------
class ResMLP(BaseModel):
    def __init__(self, structure_params, ver, input_name, task, dict_encodings,
                 pretrained_model_name="random", is_visit_based=False, output_hxg=False, dict_output_classes=None):
        super(ResMLP, self).__init__(ver, input_name, task, dict_encodings,
                                     pretrained_model_name=pretrained_model_name, is_visit_based=is_visit_based,
                                     output_hxg=output_hxg, dict_output_classes=dict_output_classes)

        self.structure_params = structure_params

    def body(self, tensor):
        # initialize with class variables
        n_layers, n_units, use_dropout, use_bn, use_ln = self.structure_params
        w_reg = self.w_reg

        def BypassDenseLayer(layer_idx, input_tensor, use_bn, use_dropout):
            out = Dense(n_units, kernel_regularizer=w_reg, name="dense{}".format(layer_idx + 1))(input_tensor)
            if use_bn:
                out = BatchNormalization(axis=-1, gamma_regularizer=w_reg, beta_regularizer=w_reg,
                                       name="bn{}".format(layer_idx + 1))(out)
            out = Activation("relu", name="act{}".format(layer_idx + 1))(out)
            out = Add()([out, input_tensor])

            if use_ln:
                out = LayerNormalization(name='ln{}'.format(layer_idx + 1))(out)
            if use_dropout:
                out = Dropout(0.5, name="dropout{}".format(layer_idx + 1))(out)

            return out

        x = tensor
        for layer_idx in range(n_layers):

            if layer_idx == 0:
                x = Dense(n_units, kernel_regularizer=w_reg, name="dense{}".format(layer_idx + 1))(x)
            else:
                x = BypassDenseLayer(layer_idx, x, use_bn, use_dropout)

        return x

# --------------------------------------
# LayerNormalization class
# --------------------------------------
class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape