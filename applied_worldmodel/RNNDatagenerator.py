import keras
import numpy as np
import multiprocessing


# TODO make one Datagenerator class to handle both VAE and RNN
def train_set_counter(file_id):
    new_data = np.load("./data/rnn_output_car_racing_" + str(file_id) + ".npz")["arr_0"]
    data = np.array([obs for obs in new_data])
    return data.shape[0]


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class RNNDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    
    def __init__(self, num_files, set_type = "debug", batch_size = 64, shuffle = True):
        """Initialization"""
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.num_files = num_files
        self.set_type = set_type
        if set_type == "debug": print("Building statistics")
        self.statistics = self.__build_statistics(set_type).T
        self.total_frame_count = sum(self.statistics.T[1])
        if set_type == "debug":
            print("Building mapping")
        
        self.mapping = None
        self.on_epoch_end()
        
        self.current_input = None
        self.current_output = None
        self.current_file_index = -1
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.total_frame_count / self.batch_size))
    
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        
        input_batch = []
        output_batch = []
        for description in self.mapping[index]:
            if self.current_file_index == description[0]:
                input_batch.extend(self.current_input[description[1]: description[2]])
                output_batch.extend(self.current_output[description[1]: description[2]])
            
            else:
                self.current_input = None
                self.current_output = None
                if self.set_type == "valid":
                    input_file = np.load("./data/rnn_input_car_racing_valid.npz")["arr_0"]
                    output_file = np.load("./data/rnn_output_car_racing_valid.npz")["arr_0"]
                else:
                    input_file = np.load("./data/rnn_input_car_racing_" + str(description[0]) + ".npz")["arr_0"]
                    output_file = np.load("./data/rnn_output_car_racing_" + str(description[0]) + ".npz")["arr_0"]
                
                self.current_input = np.array([obs for obs in input_file])
                input_batch.extend(self.current_input[description[1]: description[2]])
                
                self.current_output = np.array([obs for obs in output_file])
                output_batch.extend(self.current_output[description[1]: description[2]])
                
                self.current_file_index = description[0]
        
        ret_input = np.array(input_batch)
        ret_output = np.array(output_batch)
        
        # print(ret_input.shape)
        
        return ret_input, ret_output
    
    def __build_statistics(self, set_type):
        
        if set_type == "train":
            # count the train set using multiple threads so it"s faster and the memory will be freed
            p = multiprocessing.Pool(5)
            results = p.map(train_set_counter, range(self.num_files))
            p.terminate()
            p.join()
            
            array = np.zeros(shape = (2, self.num_files), dtype = "int")
            array[0] = range(self.num_files)
            array[1] = results
            
            return array
        
        elif set_type == "valid":
            new_data = np.load("./data/rnn_output_car_racing_valid.npz")["arr_0"]
            data = np.array([obs for obs in new_data])
            return np.array([[0], [data.shape[0]]])
        
        else:
            raise AttributeError
    
    def on_epoch_end(self):
        """Updates statistics after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.statistics)
            self.mapping = self.__build_mapping()
    
    
    def __build_mapping(self, debug = False):
        
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
            
            elif file_end and not batch_end:
                if debug: print("file_end and not batch_end")
                
                descriptions.append((self.statistics[stat_index][0], start_frame_index, file_frame_counter + 1))
                if stat_index == len(self.statistics) - 1:
                    mapping.append(descriptions)
                
                stat_index += 1
                start_frame_index = 0
                file_frame_counter = 0
                batch_frame_counter += 1
            
            elif file_end and batch_end:
                if debug: print("file_end and batch_end")
                
                descriptions.append((self.statistics[stat_index][0], start_frame_index, file_frame_counter + 1))
                mapping.append(descriptions)
                descriptions = []
                
                batch_frame_counter = 0
                stat_index += 1
                start_frame_index = 0
                file_frame_counter = 0
            
            elif not file_end and not batch_end:
                if debug: print("not file_end and not batch_end")
                
                file_frame_counter += 1
                batch_frame_counter += 1
            
            total_frame_counter += 1
        
        if debug:
            for desc in mapping:
                print(desc)
        
        return np.array(mapping)  # [[(fileindex, start_frame_index, end_frame_index), ])] interval: [start, end)


if __name__ == "__main__":
    d = RNNDataGenerator(10, "train", batch_size = 32)
    print(d.total_frame_count)
    print(d.statistics)
    print(d.mapping)
    for i in range(d.__len__()):
        d.__getitem__(i)
