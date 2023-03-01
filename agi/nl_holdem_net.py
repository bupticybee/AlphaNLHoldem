from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.utils import try_import_tf
import numpy as np
from ray.rllib.models.model import restore_original_dimensions, flatten

from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Add, Input, Flatten
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

tf = try_import_tf()

class NlHoldemNet(TFModelV2):
    """Generic vision network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        #print("obs space in ChessNet:",obs_space)
        super(NlHoldemNet, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)
        
        input_card_info = tf.keras.layers.Input(
            shape=(4, 13, 6), name="card_info")
        
        input_action_info = tf.keras.layers.Input(
            shape=(4, 5, 25), name="action_info")
        
        input_extra_info = tf.keras.layers.Input(
            shape=(2,), name="extra_info")
        
        # card conv
        x = input_card_info
        for i, (out_size, kernel, stride, blocks) in enumerate([
            [16, (3,3), 1, 1],
            [32, (3,3), 2, 2],
            [64, (3,3), 2, 2],
        ]):
            for block in range(blocks):
                subsampling = stride > 1
                y = Conv2D(out_size, kernel_size=kernel, padding="same", strides=stride, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),name="card_conv{}b{}".format(i,block))(x)
                #y = BatchNormalization()(y)
                y = Activation(tf.nn.relu)(y)
                y = Conv2D(out_size, kernel_size=kernel, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),name="card_conv{}b{}_2".format(i,block))(y)
                #y = BatchNormalization()(y)        
                if subsampling and i > 1:
                    x = Conv2D(out_size, kernel_size=(1, 1), strides=(2, 2), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),name="card_conv{}b{}_s".format(i,block))(x)
                if i > 1:
                    x = Add()([x, y])
                else:
                    x = y
                x = Activation(tf.nn.relu)(x) 
        
        x = Flatten()(x)
        last_layer_card = x
        
        # action conv
        x = input_action_info
        for i, (out_size, kernel, stride, blocks) in enumerate([
            [16, (3,3), 1, 1],
            [32, (3,3), 2, 2],
            [64, (3,3), 2, 2],
        ]):
            for block in range(blocks):
                subsampling = stride > 1
                y = Conv2D(out_size, kernel_size=kernel, padding="same", strides=stride, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),name="history_conv{}b{}".format(i,block))(x)
                #y = BatchNormalization()(y)
                y = Activation(tf.nn.relu)(y)
                y = Conv2D(out_size, kernel_size=kernel, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),name="history_conv{}b{}_2".format(i,block))(y)
                #y = BatchNormalization()(y)        
                if subsampling and i > 1:
                    x = Conv2D(out_size, kernel_size=(1, 1), strides=(2, 2), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),name="history_conv{}b{}_s".format(i,block))(x)
                if i > 1:
                    x = Add()([x, y])
                else:
                    x = y
                x = Activation(tf.nn.relu)(x) 
        
        x = Flatten()(x)
        last_layer_history = x
        
        last_layer_extra = tf.keras.layers.Dense(
            16,
            name="extra_fc",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01))(last_layer_card)
        
        feature_fuse = tf.keras.layers.Concatenate(axis=-1)([last_layer_card,last_layer_history,last_layer_extra])
        
        fc_out = tf.keras.layers.Dense(
            256,
            name="fc_1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01))(feature_fuse)
        
        fc_out = tf.keras.layers.Dense(
            128,
            name="fc_2",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01))(fc_out)
        
        fc_out = tf.keras.layers.Dense(
            64,
            name="fc_3",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(0.01))(fc_out)
        
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(fc_out)
            
        conv_out = tf.keras.layers.Dense(
            5,
            name="conv_fuse",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(
                fc_out
        )

        self.base_model = tf.keras.Model([input_card_info,input_action_info,input_extra_info], [conv_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):

        # explicit cast to float32 needed in eager
        model_out, self._value_out = self.base_model([
            tf.cast(input_dict["obs"]["card_info"], tf.float32),
            tf.cast(input_dict["obs"]["action_info"], tf.float32),
            tf.cast(input_dict["obs"]["extra_info"], tf.float32),
        ]
        )
        
        action_mask = tf.cast(input_dict["obs"]["legal_moves"], tf.float32)
        
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return model_out + inf_mask, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])