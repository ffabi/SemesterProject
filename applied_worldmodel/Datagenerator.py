# TODO
# import keras
# import numpy as np
#
# # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# class DataGenerator(keras.utils.Sequence):
#     """Generates data for Keras"""
#
#     def __init__(self, dataset, dim, batch_size =10, shuffle =True):
#         """Initialization"""
#         self.dim = dim
#         self.batch_size = batch_size
#         self.dataset = dataset
#         self.shuffle = shuffle
#         self.indexes = np.arange(len(dataset))
#         self.on_epoch_end()
#
#
#     def __len__(self):
#         """Denotes the number of batches per epoch"""
#         return int(np.ceil(len(self.dataset) / self.batch_size))
#
#     def __getitem__(self, index):
#         """Generate one batch of data"""
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
#
#         ids = [self.dataset["frame"][k] for k in indexes]
#         datapoints = [self.dataset["data"][k] for k in indexes]
#
#         """Generates data containing batch_size samples"""
#         # Initialization
#
#         X = np.empty((self.batch_size, *self.dim), dtype =float)
#         Y = np.empty((self.batch_size, *(1, )), dtype =float)
#         # Generate data
#         for i, ID in enumerate(ids):
#             img = img_to_array(load_img('dataset/resized_frames/frame_' + str(ID) + '.png')) /255.0
#
#             X[i,] = img
#             Y[i,] = datapoints[i]
#
#         return X, Y
#
#     def on_epoch_end(self):
#         """Updates indexes after each epoch"""
#         if self.shuffle:
#             np.random.shuffle(self.indexes)