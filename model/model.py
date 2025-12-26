#!/usr/bin/env python

'''
Architecture of a variational autoencoder model.
'''

from __future__ import print_function
import json
import keras
from keras import ops, layers
from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, Layer
from keras.models import Model, model_from_json
from keras.losses import binary_crossentropy
from keras.saving import register_keras_serializable

@register_keras_serializable()
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

@register_keras_serializable()
class VAELossLayer(Layer):
    def call(self, inputs):
        x, x_decoded, z_mean, z_log_var = inputs
        
        shape = ops.shape(x)
        if keras.config.image_data_format() == 'channels_first':
            n_pixels = shape[2] * shape[3]
        else:
            n_pixels = shape[1] * shape[2]
        n_pixels = ops.cast(n_pixels, 'float32')

        # Flatten for loss calculation
        x_flat = ops.reshape(x, (shape[0], -1))
        x_decoded_flat = ops.reshape(x_decoded, (shape[0], -1))

        # Reconstruction Loss
        xent_loss = n_pixels * binary_crossentropy(x_flat, x_decoded_flat)

        # KL Divergence
        kl_loss = -0.5 * ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)

        # Add loss to the graph
        self.add_loss(ops.mean(xent_loss + kl_loss))
        
        # Return the decoded image so the model output remains standard
        return x_decoded

class Vae(object):
    def __init__(self, latent_dim=64, img_dim=(3, 64, 64), batch_size=100,
         intermediate_dim=1024, filters=64, num_conv=3):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.intermediate_dim = intermediate_dim
        self.filters, self.num_conv = filters, num_conv
        self.img_chns, self.img_cols, self.img_rows = img_dim

        if keras.config.image_data_format() == 'channels_first':
            self.original_img_size = (self.img_chns, self.img_rows, self.img_cols)
        else:
            self.original_img_size = (self.img_rows, self.img_cols, self.img_chns)

    def build_model(self, mpath):
        # Config
        img_chns, img_cols, img_rows = self.img_chns, self.img_cols, self.img_rows
        latent_dim = self.latent_dim
        intermediate_dim = self.intermediate_dim
        filters, num_conv = self.filters, self.num_conv
        
        # Encoder
        x = Input(shape=self.original_img_size, name='input')
        
        # Convolutional Block
        conv_1 = Conv2D(img_chns, kernel_size=(2, 2), padding='same', activation='relu', name='conv_1')(x)
        conv_2 = Conv2D(filters, kernel_size=(2, 2), padding='same', activation='relu', name='conv_2', strides=(2, 2))(conv_1)
        conv_3 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', name='conv_3', strides=1)(conv_2)
        conv_4 = Conv2D(filters, kernel_size=num_conv, padding='same', activation='relu', name='conv_4', strides=1)(conv_3)
        
        flat = Flatten(name='flat')(conv_4)
        hidden = Dense(intermediate_dim, activation='relu', name='hidden')(flat)

        z_mean = Dense(latent_dim, name='z_mean')(hidden)
        z_log_var = Dense(latent_dim, name='z_log_var')(hidden)

        # Use Custom Sampling Layer
        z = Sampling(name='sampling')([z_mean, z_log_var])

        # Decoder
        up_dim = img_rows // 2
        decoder_hid = Dense(intermediate_dim, activation='relu', name='decoder_hid')
        decoder_upsample = Dense(filters * up_dim * up_dim, activation='relu', name='decoder_upsample')

        if keras.config.image_data_format() == 'channels_first':
            output_shape = (self.batch_size, filters, up_dim, up_dim)
        else:
            output_shape = (self.batch_size, up_dim, up_dim, filters)

        decoder_reshape = Reshape(output_shape[1:], name='decoder_reshape')
        decoder_deconv_1 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu', name='decoder_deconv_1')
        decoder_deconv_2 = Conv2DTranspose(filters, kernel_size=num_conv, padding='same', strides=1, activation='relu', name='decoder_deconv_2')
        decoder_deconv_3_upsamp = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu', name='decoder_deconv_3_upsamp')
        decoder_mean_squash = Conv2D(img_chns, kernel_size=2, padding='valid', activation='sigmoid', name='decoder_mean_squash')

        hid_decoded = decoder_hid(z)
        up_decoded = decoder_upsample(hid_decoded)
        reshape_decoded = decoder_reshape(up_decoded)
        deconv_1_decoded = decoder_deconv_1(reshape_decoded)
        deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
        x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
        x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

        final_output = VAELossLayer(name='vae_loss')([x, x_decoded_mean_squash, z_mean, z_log_var])

        # Instantiate Model
        vae = Model(x, final_output)
        
        # Compile with loss=None because loss is added inside VAELossLayer
        vae.compile(optimizer="rmsprop")

        # Save model JSON
        with open(mpath, 'w') as outfile:
            json.dump(vae.to_json(), outfile)

        #Sub-models
        encoder = Model(x, z_mean)
        
        decoder_input = Input(shape=(latent_dim,))
        _hid_decoded = decoder_hid(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
        decoder = Model(decoder_input, _x_decoded_mean_squash)

        return vae, encoder, decoder

    def read(self, fn, weights):
        with open(fn, 'r') as infile:
            json_str = json.load(infile)
            vae = model_from_json(json_str)
            
        vae.load_weights(weights)
        vae.compile(optimizer='rmsprop')

        # Recover sub-models using get_layer
        encoder = self.recover_encoder(vae)
        decoder = self.recover_decoder(vae)

        return vae, encoder, decoder

    def recover_encoder(self, vae):
        x = vae.get_layer('input').input
        z_mean = vae.get_layer('z_mean').output
        return Model(x, z_mean)

    def recover_decoder(self, vae):
        decoder_hid = vae.get_layer('decoder_hid')
        decoder_upsample = vae.get_layer('decoder_upsample')
        decoder_reshape = vae.get_layer('decoder_reshape')
        decoder_deconv_1 = vae.get_layer('decoder_deconv_1')
        decoder_deconv_2 = vae.get_layer('decoder_deconv_2')
        decoder_deconv_3_upsamp = vae.get_layer('decoder_deconv_3_upsamp')
        decoder_mean_squash = vae.get_layer('decoder_mean_squash')

        decoder_input = Input(shape=(self.latent_dim,))
        _hid_decoded = decoder_hid(decoder_input)
        _up_decoded = decoder_upsample(_hid_decoded)
        _reshape_decoded = decoder_reshape(_up_decoded)
        _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
        _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
        _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
        _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
        return Model(decoder_input, _x_decoded_mean_squash)

    def to_image(self, array):
        array = array.reshape(self.img_rows, self.img_cols, self.img_chns)
        array *= 255
        return array.astype('uint8')

    def init_model(self, infile, weights):
        try:
            vae, encoder, decoder = self.read(infile, weights)
            print('Successfully loaded model: {}'.format(weights))
        except Exception as e:
            print(f'Loading failed ({e}). Instantiating new models ...')
            vae, encoder, decoder = self.build_model(infile)

        return vae, encoder, decoder