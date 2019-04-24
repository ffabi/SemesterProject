import math, datetime
import numpy as np

from keras.layers import Input, LSTM, Dense, TimeDistributed, Activation, Flatten
from keras.models import Model
from keras import backend as K, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard, LambdaCallback
from keras.optimizers import Adam

from rnn.RNNDataGenerator import *

Z_DIM = 32
ACTION_DIM = 3

GAUSSIAN_MIXTURES = 3


def get_mixture_coef(y_pred):
    d = GAUSSIAN_MIXTURES * Z_DIM

    rollout_length = K.shape(y_pred)[1]

    pi = y_pred[:, :, :d]
    mu = y_pred[:, :, d:(2 * d)]
    log_sigma = y_pred[:, :, (2 * d):(3 * d)]
    # discrete = y_pred[:,3*GAUSSIAN_MIXTURES:]

    pi = K.reshape(pi, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    mu = K.reshape(mu, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    log_sigma = K.reshape(log_sigma, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])

    pi = K.exp(pi) / K.sum(K.exp(pi), axis=2, keepdims=True)
    sigma = K.exp(log_sigma)

    return pi, mu, sigma  # , discrete


def tf_normal(y_true, mu, sigma, pi):
    rollout_length = K.shape(y_true)[1]
    y_true = K.tile(y_true, (1, 1, GAUSSIAN_MIXTURES))
    y_true = K.reshape(y_true, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])

    oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)
    result = y_true - mu
    #   result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result) / 2
    result = K.exp(result) * (1 / (sigma + 1e-8)) * oneDivSqrtTwoPI
    result = result * pi
    result = K.sum(result, axis=2)  #### sum over gaussians
    # result = K.prod(result, axis=2) #### multiply over latent dims
    return result


class RNN:
    def __init__(self):
        self.z_dim = Z_DIM
        self.action_dim = ACTION_DIM
        self.hidden_units = 300
        self.gaussian_mixtures = GAUSSIAN_MIXTURES
        self.input_dim = (self.z_dim + self.action_dim)
        self.learning_rate = 0.0001
        self.batch_size = 47
        self.epochs = 1000
        self.kl_divider = 32
        # self.seq_length = 199
        self.type = "original"
        self.loss = "both"
        self.activation = "tanh"
        self.name = "RNN-type:{}-date:{}-action_dim:{}-hidden_units:{}-gaussian_mixtures:{}-learning_rate:{}-batch_size:{}-loss:{}-kl_divider:{}-activation:{}".format(
            self.type, str(datetime.datetime.now())[:16], self.action_dim, self.hidden_units, self.gaussian_mixtures,
            self.learning_rate, self.batch_size, self.loss, self.kl_divider, self.activation)

        if self.type == "original":
            self.models = self._build_original()
        elif self.type == "simple_lstm":
            self.models = self._build_simple_lstm()
        else:
            raise ValueError("Invalid network type")

        self.model = self.models[0]
        self.forward = self.models[1]

    def _build_original(self):
        #### THE MODEL THAT WILL BE TRAINED
        rnn_x = Input(shape=(None, self.z_dim + self.action_dim))
        lstm = LSTM(self.hidden_units, return_sequences=True, return_state=True, activation=self.activation)

        lstm_output, _, _ = lstm(rnn_x)
        mdn = Dense(GAUSSIAN_MIXTURES * (3 * Z_DIM))(lstm_output)  # + discrete_dim
        rnn = Model(rnn_x, mdn)

        #### THE MODEL USED DURING PREDICTION
        state_input_h = Input(shape=(self.hidden_units,))
        state_input_c = Input(shape=(self.hidden_units,))
        state_inputs = [state_input_h, state_input_c]
        _, state_h, state_c = lstm(rnn_x, initial_state=state_inputs)

        forward = Model([rnn_x] + state_inputs, [state_h, state_c])

        #### LOSS FUNCTION

        def rnn_r_loss(y_true, y_pred):
            pi, mu, sigma = get_mixture_coef(y_pred)

            result = tf_normal(y_true, mu, sigma, pi)

            result = -K.log(result + 1e-8)
            result = K.mean(result, axis=(1, 2))  # mean over rollout length and z dim

            return result

        def rnn_kl_loss(y_true, y_pred):
            pi, mu, sigma = get_mixture_coef(y_pred)
            kl_loss = - 0.5 * K.mean(1 + K.log(K.square(sigma)) - K.square(mu) - K.square(sigma), axis=[1, 2, 3])
            return kl_loss

        def rnn_loss(y_true, y_pred):
            if self.loss == "r_only":
                return rnn_r_loss(y_true, y_pred)  # + rnn_kl_loss(y_true, y_pred)
            elif self.loss == "both":
                return rnn_r_loss(y_true, y_pred) + rnn_kl_loss(y_true, y_pred) / self.kl_divider

        rnn.compile(loss=rnn_loss, optimizer=Adam(), metrics=[rnn_r_loss, rnn_kl_loss])
        # rnn.summary()

        return rnn, forward


    def _build_simple_lstm(self):
        rnn_x = Input(shape=(None, self.z_dim + self.action_dim))
        lstm = LSTM(self.hidden_units, return_sequences=True, return_state=False, activation=self.activation)
        lstm_output = lstm(rnn_x)
        mdn = Dense(self.z_dim)(lstm_output)

        rnn_model = Model(rnn_x, mdn)
        rnn_model.compile(loss="mse", optimizer=Adam())

        return rnn_model, rnn_model

    # def _build_simple_lstm(self):
        # rnn_x = Input(shape=(None, self.z_dim + self.action_dim))
        # lstm = LSTM(self.hidden_units, return_sequences=True, return_state=False, activation=self.activation)
        # lstm_output = lstm(rnn_x)
        # mdn = Dense(self.z_dim)(lstm_output)
        #
        # rnn_model = Model(rnn_x, mdn)
        # rnn_model.compile(loss="mse", optimizer=Adam())
        #
        # return rnn_model, rnn_model


    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, num_files):

        train_generator = RNNDataGenerator(
            num_files=num_files,
            set_type="train",
            batch_size=self.batch_size,
            shuffle=True,
        )

        validation_generator = RNNDataGenerator(
            num_files=0,
            set_type="valid",
            batch_size=self.batch_size,
            shuffle=True,
        )

        earlystop = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00001,
            patience=5,
            verbose=1,
            mode='auto',
        )

        # we are going to keep only the best model
        mcp = ModelCheckpoint(
            filepath='./rnn/weights_' + self.name + '.h5',
            verbose=1,
            save_best_only=True,
        )

        ton = TerminateOnNaN()

        tensorboard = TensorBoard(
            log_dir='./log/{}'.format(self.name),
            write_graph=False,
            write_grads=False,
            write_images=False,
        )

        lambda_callback = LambdaCallback(
            on_epoch_begin=None,
            on_epoch_end=None,

            on_batch_begin=None,
            on_batch_end=None,

            on_train_begin=None,
            on_train_end=None
        )

        callbacks_list = [tensorboard, earlystop, mcp, ton, lambda_callback]

        print(self.name)

        self.model.summary()

        self.model.fit_generator(
            generator=train_generator,
            validation_data=validation_generator,
            use_multiprocessing=False,
            shuffle=False,
            epochs=self.epochs,
            max_queue_size=256,
            callbacks=callbacks_list
        )

        # self.model.fit(rnn_input, rnn_output,
        #                shuffle=True,
        #                epochs=20,
        #                batch_size=32,
        #                validation_split=validation_split,
        #                callbacks=callbacks_list)

        # self.model.save_weights('./rnn/weights.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)


if __name__ == '__main__':
    rnn = RNN()
    rnn.model.summary()
