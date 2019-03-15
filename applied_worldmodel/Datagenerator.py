# TODO
import keras
import numpy as np
import multiprocessing


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class oldDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, dataset, dim, batch_size = 10, shuffle =True):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()


    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        
        # TODO
        
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        ids = [self.dataset["frame"][k] for k in indexes]
        datapoints = [self.dataset["data"][k] for k in indexes]

        """Generates data containing batch_size samples"""
        # Initialization

        X = np.empty((self.batch_size, *self.dim), dtype =float)
        Y = np.empty((self.batch_size, *(1, )), dtype =float)
        # Generate data
        for i, ID in enumerate(ids):
            img = img_to_array(load_img('dataset/resized_frames/frame_' + str(ID) + '.png')) /255.0

            X[i,] = img
            Y[i,] = datapoints[i]

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
            
            
def train_set_counter(file_id):
    new_data = np.load('./data/obs_data_car_racing_' + str(file_id) + '.npz')["arr_0"]
    data = np.array([item for obs in new_data for item in obs])
    return data.shape[0]

    
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    
    def __init__(self, max_batch, set_type = "debug", batch_size = 64, shuffle = True):
        """Initialization"""

        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.num_files = max_batch
        
        self.statistics = np.array(self.__count_all_frames__(set_type).T)
        self.total_frame_count = sum(self.statistics)
        np.random.shuffle(self.statistics)
        # print(self.statistics)
        
        # self.statistics = self.statistics/64
        #
        # for i in range(1, len(self.statistics)):
        #     self.statistics[i] += self.statistics[i-1]
        
        

    def __count_all_frames__(self, set_type):
        if set_type == "debug":
            return np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                    [12592, 12356, 12582, 12342, 12421, 12325, 12487, 12527, 12152, 12383, 12372, 12325, 12352, 12442, 12230, 12530, 12458, 12373, 12482]])
        
        elif set_type== "train":
            #count the train set using a different thread so the memory will be freed
            p = multiprocessing.Pool(3)
            results = p.map(train_set_counter, range(self.num_files))
            p.terminate()
            p.join()
            
            array = np.zeros(shape = (2, self.num_files), dtype = "int")
            array[0] = range(self.num_files)
            array[1] = results
            
            return array
        
        elif set_type== "valid":
            new_data = np.load('./data/obs_data_car_racing_valid.npz')["arr_0"]
            data = np.array([item for obs in new_data for item in obs])
            return data.shape[0]
    

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.total_frame_count / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        ids = [self.dataset["frame"][k] for k in indexes]
        datapoints = [self.dataset["data"][k] for k in indexes]
        
        """Generates data containing batch_size samples"""
        # Initialization
        
        X = np.empty((self.batch_size, *self.dim), dtype = float)
        Y = np.empty((self.batch_size, *(1,)), dtype = float)
        # Generate data
        for i, ID in enumerate(ids):
            img = img_to_array(load_img('dataset/resized_frames/frame_' + str(ID) + '.png')) / 255.0
            
            X[i,] = img
            Y[i,] = datapoints[i]
        
        return X, Y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.statistics)
    
    
    
if __name__ == '__main__':
    d = DataGenerator(19)
    print(d.statistics)