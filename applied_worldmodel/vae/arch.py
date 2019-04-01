import numpy as np

from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape, BatchNormalization
from keras.models import Model, Sequential
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, TerminateOnNaN, LambdaCallback
from keras.optimizers import Adam
import datetime
import pickle

from Datagenerator import DataGenerator

INPUT_DIM = (64,64,3)

CONV_FILTERS = [32,64,128,256]
CONV_KERNEL_SIZES = [4,4,4,4]
CONV_STRIDES = [2,2,2,2]
CONV_ACTIVATIONS = ['relu','relu','relu','relu']

DENSE_SIZE = 1024

CONV_T_FILTERS = [128,64,32,3]
CONV_T_KERNEL_SIZES = [5,5,6,6]
CONV_T_STRIDES = [2,2,2,2]
CONV_T_ACTIVATIONS = ['relu','relu','relu','sigmoid']

Z_DIM = 32

EPOCHS = 20
BATCH_SIZE = 19

LEARNING_RATE = 0.0004

KL_DIVIDER = 0

BETA = 0.69

NAME = "VAE-original-date:{}-KL_DIVIDER:{}-BATCH_SIZE:{}-LEARNING_RATE:{}-BETA:{}".format(str(datetime.datetime.now())[:16], KL_DIVIDER, BATCH_SIZE, LEARNING_RATE, BETA)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0.,stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

class VAE:
    def __init__(self):
        self.model_parameters = self._build()
        self.model = self.model_parameters[0]
        self.encoder = self.model_parameters[1]
        self.decoder = self.model_parameters[2]
        
        self.vae_z_mean = self.model_parameters[3]
        self.vae_z_log_var = self.model_parameters[4]

        self.input_dim = INPUT_DIM
        self.z_dim = Z_DIM

        self.kl_divider = KL_DIVIDER
        

    def loss_generator(self):
    
        def vae_r_loss(y_true, y_pred):
        
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
        
            # return K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)
            return K.mean(K.abs(y_true_flat - y_pred_flat), axis = -1)
    
    
        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + self.vae_z_log_var - K.square(self.vae_z_mean) - K.exp(self.vae_z_log_var), axis = -1)
    
    
        def vae_loss(y_true, y_pred):
            if self.kl_divider == 0:
                return vae_r_loss(y_true, y_pred)
            else:
                return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred) / self.kl_divider
            
        return vae_r_loss, vae_kl_loss, vae_loss


    def _build_old(self):
        vae_x = Input(shape=INPUT_DIM)
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0])(vae_x)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0])(vae_c1)
        vae_c3= Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0])(vae_c2)
        vae_c4= Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0])(vae_c3)

        vae_z_in = Flatten()(vae_c4)

        vae_z_mean = Dense(Z_DIM)(vae_z_in)
        vae_z_log_var = Dense(Z_DIM)(vae_z_in)

        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])

        vae_z_input = Input(shape=(Z_DIM,))

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z)

        vae_z_out = Reshape((1,1,DENSE_SIZE))
        vae_z_out_model = vae_z_out(vae_dense_model)

        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0])
        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1])
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2])
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3])
        vae_d4_model = vae_d4(vae_d3_model)

        #### DECODER ONLY

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        #### MODELS

        vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        return vae, vae_encoder, vae_decoder, vae_z_mean, vae_z_log_var
    
    
    def _build(self):
        vae_input = Input(shape=INPUT_DIM)
        vae_c1 = Conv2D(filters = CONV_FILTERS[0], kernel_size = CONV_KERNEL_SIZES[0], strides = CONV_STRIDES[0], activation=CONV_ACTIVATIONS[0])(vae_input)
        vae_c1_norm = BatchNormalization()(vae_c1)
        vae_c2 = Conv2D(filters = CONV_FILTERS[1], kernel_size = CONV_KERNEL_SIZES[1], strides = CONV_STRIDES[1], activation=CONV_ACTIVATIONS[0])(vae_c1_norm)
        vae_c2_norm = BatchNormalization()(vae_c2)
        vae_c3 = Conv2D(filters = CONV_FILTERS[2], kernel_size = CONV_KERNEL_SIZES[2], strides = CONV_STRIDES[2], activation=CONV_ACTIVATIONS[0])(vae_c2_norm)
        vae_c3_norm = BatchNormalization()(vae_c3)
        vae_c4 = Conv2D(filters = CONV_FILTERS[3], kernel_size = CONV_KERNEL_SIZES[3], strides = CONV_STRIDES[3], activation=CONV_ACTIVATIONS[0])(vae_c3_norm)
        vae_c4_norm = BatchNormalization()(vae_c4)

        vae_z_in = Flatten()(vae_c4_norm)
        vae_z_in_norm = BatchNormalization()(vae_z_in)

 
        vae_z_mean = Dense(Z_DIM)(vae_z_in_norm)
        vae_z_mean_norm = BatchNormalization()(vae_z_mean)
        vae_z_log_var = Dense(Z_DIM)(vae_z_in_norm)
        vae_z_log_var_norm = BatchNormalization()(vae_z_log_var)

        vae_z = Lambda(sampling)([vae_z_mean_norm, vae_z_log_var_norm])
        vae_z_norm = BatchNormalization()(vae_z)

        vae_z_input = Input(shape=(Z_DIM,))

        # we instantiate these layers separately so as to reuse them later
        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z_norm)
        vae_dense_model_norm = BatchNormalization()(vae_dense_model)

        vae_z_out = Reshape((1,1,DENSE_SIZE))
        vae_z_out_model = vae_z_out(vae_dense_model_norm)
        vae_z_out_model_norm = BatchNormalization()(vae_z_out_model)

        vae_d1 = Conv2DTranspose(filters = CONV_T_FILTERS[0], kernel_size = CONV_T_KERNEL_SIZES[0] , strides = CONV_T_STRIDES[0], activation=CONV_T_ACTIVATIONS[0])
        vae_d1_model = vae_d1(vae_z_out_model_norm)
        vae_d1_model_norm = BatchNormalization()(vae_d1_model)
        vae_d2 = Conv2DTranspose(filters = CONV_T_FILTERS[1], kernel_size = CONV_T_KERNEL_SIZES[1] , strides = CONV_T_STRIDES[1], activation=CONV_T_ACTIVATIONS[1])
        vae_d2_model = vae_d2(vae_d1_model_norm)
        vae_d2_model_norm = BatchNormalization()(vae_d2_model)
        vae_d3 = Conv2DTranspose(filters = CONV_T_FILTERS[2], kernel_size = CONV_T_KERNEL_SIZES[2] , strides = CONV_T_STRIDES[2], activation=CONV_T_ACTIVATIONS[2])
        vae_d3_model = vae_d3(vae_d2_model_norm)
        vae_d3_model_norm = BatchNormalization()(vae_d3_model)
        vae_d4 = Conv2DTranspose(filters = CONV_T_FILTERS[3], kernel_size = CONV_T_KERNEL_SIZES[3] , strides = CONV_T_STRIDES[3], activation=CONV_T_ACTIVATIONS[3])
        vae_d4_model = vae_d4(vae_d3_model_norm)
        vae_d4_model_norm = BatchNormalization()(vae_d4_model)

        #### DECODER ONLY

        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)

        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d2(vae_d1_decoder)
        vae_d3_decoder = vae_d3(vae_d2_decoder)
        vae_d4_decoder = vae_d4(vae_d3_decoder)

        #### MODELS

        vae = Model(vae_input, vae_d4_model_norm)
        vae_encoder = Model(vae_input, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        return vae, vae_encoder, vae_decoder, vae_z_mean, vae_z_log_var


    def set_weights(self, filepath):
        self.model.load_weights(filepath)
        
    def on_epoch_end(self, epoch, logs):
        if self.kl_divider == 0:
            self.kl_divider = 32768
        else:
            self.kl_divider *= BETA
            if epoch >= 10:
                self.kl_divider = 1024
        print(self.kl_divider)

        vae_r_loss, vae_kl_loss, vae_loss = self.loss_generator()
        self.model.compile(optimizer=Adam(lr = LEARNING_RATE), loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

    def train(self, num_files):
        
        vae_r_loss, vae_kl_loss, vae_loss = self.loss_generator()
        self.model.compile(optimizer=Adam(lr = LEARNING_RATE), loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])


        train_generator = DataGenerator(
            num_files = num_files,
            set_type = "train",
            batch_size = BATCH_SIZE,
            shuffle = True,
        )

        validation_generator = DataGenerator(
            num_files = 0,
            set_type = "valid",
            batch_size = BATCH_SIZE,
            shuffle = True,
        )

        earlystop = EarlyStopping(
            monitor = 'val_loss',
            min_delta = 0.0001,
            patience = 5,
            verbose = 1,
            mode = 'auto',
        )

        # we are going to keep only the best model
        mcp = ModelCheckpoint(
            filepath = './vae/weights_' + NAME + '.h5',
            verbose = 1,
            save_best_only = True,
        )

        
        ton = TerminateOnNaN()

        tensorboard = TensorBoard(
            log_dir='./log/{}'.format(NAME),
            write_graph=False,
            write_grads=False,
            write_images=False,
        )
        
        lambda_callback = LambdaCallback(
            on_epoch_begin=None,
            on_epoch_end=self.on_epoch_end,
            
            on_batch_begin=None,
            on_batch_end=None,
            
            on_train_begin=None,
            on_train_end=None
        )
        
        callbacks_list = [tensorboard, earlystop, mcp, ton, lambda_callback]
        
        print(NAME)

        self.model.fit_generator(
            generator = train_generator,
            validation_data = validation_generator,
            use_multiprocessing = False,
            shuffle=False,
            epochs=EPOCHS,
            max_queue_size = 128,
            callbacks=callbacks_list
        )
        
        # with open('./log/history_' + NAME, 'wb') as pickle_file:
        #     pickle.dump(history, pickle_file)
        
        # self.model.save_weights('./vae/weights.h5')
        
    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def generate_rnn_data(self, obs_data, action_data):

        rnn_input = []
        rnn_output = []

        for i, j in zip(obs_data, action_data):
            rnn_z_input = self.encoder.predict(np.array(i))
            conc = [np.concatenate([x,y]) for x, y in zip(rnn_z_input, j)]
            rnn_input.append(conc[:-1])
            rnn_output.append(np.array(rnn_z_input[1:]))

        rnn_input = np.array(rnn_input)
        rnn_output = np.array(rnn_output)

        return (rnn_input, rnn_output)
    



