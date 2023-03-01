from ray.rllib.models.tf.tf_modelv2 import TFModelV2

from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Add, Input, Flatten
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.models.tf.misc import normc_initializer, get_activation_fn
from ray.rllib.utils import try_import_tf
import numpy as np
from ray.rllib.models.model import restore_original_dimensions, flatten

tf = try_import_tf()

class ResNet(TFModelV2):
    """Generic vision network implemented in ModelV2 API."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        #print("obs space in ChessNet:",obs_space)
        super(ResNet, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        activation = get_activation_fn(model_config.get("conv_activation"))
        filters = model_config.get("conv_filters")
        if not filters:
            filters = _get_filter_config(obs_space.shape)
        no_final_linear = model_config.get("no_final_linear")
        vf_share_layers = model_config.get("vf_share_layers")

        inputs = tf.keras.layers.Input(
            shape=model_config["custom_options"]["size"], name="observations")
        
        
        side = tf.keras.layers.Input(
            shape=model_config["custom_options"]["side"], name="side")
        
        side_plat = tf.keras.layers.Lambda(
                lambda x: 
                    tf.broadcast_to(tf.expand_dims(tf.expand_dims(x[0],1),1),
                                   [
                                       tf.shape(x[1])[0],
                                       tf.shape(x[1])[1],
                                       tf.shape(x[1])[2],
                                       tf.shape(x[0])[1]
                                   ])
        )([side,inputs])
        
        last_layer = tf.keras.layers.Lambda(lambda x:tf.concat([x[0],x[1]],axis=-1))([inputs,side_plat])
        #last_layer = inputs

        # Build the action layers
        x = last_layer
        for i, (out_size, kernel, stride, blocks) in enumerate(filters[:-1], 1):
            """
            last_layer = tf.keras.layers.Conv2D(
                out_size,
                kernel,
                strides=(stride, stride),
                activation=activation,
                padding="same",
                name="conv{}".format(i))(last_layer)
            """
            for block in range(blocks):
                subsampling = stride > 1
                y = Conv2D(out_size, kernel_size=kernel, padding="same", strides=stride, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),name="conv{}b{}".format(i,block))(x)
                #y = BatchNormalization()(y)
                y = Activation(tf.nn.relu)(y)
                y = Conv2D(out_size, kernel_size=kernel, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),name="conv{}b{}_2".format(i,block))(y)
                #y = BatchNormalization()(y)        
                if subsampling and i > 1:
                    x = Conv2D(out_size, kernel_size=(1, 1), strides=(2, 2), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),name="conv{}b{}_s".format(i,block))(x)
                if i > 1:
                    x = Add()([x, y])
                else:
                    x = y
                x = Activation(tf.nn.relu)(x)
                
        out_size, kernel, stride,_ = filters[-1]
        
        last_layer = x
        
        last_layer = tf.keras.layers.Conv2D(
            out_size,
            kernel,
            strides=(stride, stride),
            activation=activation,
            padding="valid",
            name="conv{}".format(i + 1))(last_layer)
        conv_out = tf.keras.layers.Conv2D(
            model_config["custom_options"]["bottleneck"],
            [1, 1],
            activation=None,
            padding="same",
            name="conv_out")(last_layer)

        # Build the value layers
        last_layer = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x, axis=[1, 2]))(last_layer)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(last_layer)
            
        conv_out = tf.squeeze(conv_out, axis=[1, 2])
        conv_out = tf.keras.layers.Dense(
            num_outputs,
            name="conv_fuse",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(
                conv_out
        )

        self.base_model = tf.keras.Model([inputs,side], [conv_out, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):

        # explicit cast to float32 needed in eager
        model_out, self._value_out = self.base_model([
            tf.cast(input_dict["obs"]["board"], tf.float32),
            tf.cast(input_dict["obs"]["side"], tf.float32)]
        )
        
        action_mask = tf.cast(input_dict["obs"]["legal_moves"], tf.float32)
        
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return model_out + inf_mask, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
