import keras
import numpy as np
import multiprocessing

def train_set_counter(file_id):
    new_data = np.load('./data/obs_data_car_racing_' + str(file_id) + '.npz')["arr_0"]
    data = np.array([item for obs in new_data for item in obs])
    return data.shape[0]

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    
    def __init__(self, max_batch, set_type = "debug", batch_size = 64, shuffle = True):
        """Initialization"""
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.num_files = max_batch
        
        self.statistics = self.__count_all_frames__(set_type).T
        self.total_frame_count = sum(self.statistics.T[1])
        np.random.shuffle(self.statistics)
        self.mapping = self.build_mapping(debug = set_type == "debug")
        
        
        # self.statistics = self.statistics/64
        #
        # for i in range(1, len(self.statistics)):
        #     self.statistics[i] += self.statistics[i-1]
    
    
    
    def __count_all_frames__(self, set_type):
        if set_type == "debug":
            # return np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            #                  [12592, 12356, 12582, 12342, 12421, 12325, 12487, 12527, 12152, 12383, 12372, 12325, 12352, 12442, 12230, 12530, 12458, 12373, 12482]])
            return np.array([[0, 1],
                             [20, 20]])
        
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
            new_data = np.load('./data/obs_test.npz')["arr_0"]
            data = np.array([item for obs in new_data for item in obs])
            return np.array([[0], [data.shape[0]]])
    
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.total_frame_count / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        
        batch = []
        
        for description in self.mapping[index]:
            file = np.load('./data/obs_data_carracing_' + str(description[0]) + '.npz')["arr_0"]
            frames = np.array([item for obs in file for item in obs])
            batch.append(frames[description[1] : description[2]])
        
        return batch, batch
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.statistics)
            self.mapping = self.build_mapping()

    def build_mapping(self, debug = False):
    
        mapping = []
        descriptions = []

        file_frame_counter = 0
        batch_frame_counter = 0
        total_frame_counter = 0
        
        start_frame_index = 0
        
        stat_index = 0
        
        while total_frame_counter < self.total_frame_count and stat_index < len(self.statistics):
            
            batch_end = (batch_frame_counter == self.batch_size - 1)
            file_end = (file_frame_counter == self.statistics[stat_index][1] - 1)
            
            if batch_end and not file_end:
                if debug: print("batch_end and not file_end")
                
                descriptions.append((self.statistics[stat_index][0], start_frame_index, file_frame_counter + 1))
                mapping.append(descriptions)
                descriptions = []
    
                start_frame_index = file_frame_counter + 1
                batch_frame_counter = 0
                file_frame_counter += 1
                total_frame_counter += 1
                
            elif file_end and not batch_end:
                if debug: print("file_end and not batch_end")
    
                descriptions.append((self.statistics[stat_index][0], start_frame_index, file_frame_counter + 1))
                if stat_index == len(self.statistics) - 1:
                    mapping.append(descriptions)
    
                stat_index += 1
                start_frame_index = 0
                file_frame_counter = 0
                batch_frame_counter += 1
                total_frame_counter += 1
                
            elif file_end and batch_end:
                if debug: print("file_end and batch_end")
    
                descriptions.append((self.statistics[stat_index][0], start_frame_index, file_frame_counter + 1))
                mapping.append(descriptions)
                descriptions = []
    
                batch_frame_counter = 0
                stat_index += 1
                start_frame_index = 0
                file_frame_counter = 0
                total_frame_counter += 1
                
            elif not file_end and not batch_end:
                if debug: print("not file_end and not batch_end")

                file_frame_counter += 1
                batch_frame_counter += 1
                total_frame_counter += 1
                
        if debug:
            for d in mapping:
                print(d)
        
        return np.array(mapping)  # [[(fileindex, start_frame_index, end_frame_index), (fileindex2, start_frame_index2, end_frame_index2)])]


if __name__ == '__main__':
    d = DataGenerator(18, "debug", batch_size = 12)
    print(len(d.mapping))
    print(d.__len__())
    