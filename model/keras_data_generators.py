"""This module handles the different keras data generators"""

import copy 
import tensorflow.keras as keras
import numpy as np 

from traj_processor import TrajProcessor

class KerasFitGenerator(keras.utils.Sequence):
    """Generator for the training and validation"""
    def __init__(self, X, y, topk_weights, batch_size):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.topk_weights = topk_weights
        self.batch_size = batch_size

        # Shuffle the dataset so that when we pick the negative samples later
        # it will also be randomized
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]

        
        
    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))
        
        
    def on_epoch_end(self):
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
        
    
    def __getitem__(self, index):
        # Get current batch 
        batch_end = (index+1)*self.batch_size
        if batch_end > len(self.X):
            batch_end = len(self.X)
        X = self.X[index*self.batch_size:batch_end]
        y = self.y[index*self.batch_size:batch_end]
        
        # Get next batch. If next batch is not a full batch, get the first 
        next_batch_end = (index+2)*self.batch_size
        if next_batch_end > len(self.X):
            X_next = self.X[:self.batch_size]
        else:
            X_next_start = (index + 1) * self.batch_size
            X_next_end = (index + 2) * self.batch_size
            X_next = self.X[X_next_start : X_next_end]

        # We use the next batch to get a sample of negative trajectories. 
        # This works the same as randomizing the negative trajectories simply 
        # because we randomize the data after every epoch.
        X_neg = X_next[:len(X),0]
        X_neg = X_neg.reshape(len(X_neg), 1)
        X_1 = np.concatenate((X[:,:2], X_neg), axis=1)
        X_2 = X[:,2:]
        neg_2 = X_next[:len(X),7:]
        X_3 = np.concatenate((X_1, X_2), axis=1)
        # if X_3.shape[0] < neg_2.shape[0]:
        #     padding = np.zeros((neg_2.shape[0] - X_3.shape[0], X_3.shape[1]))
        #     X_3 = np.vstack((X_3, padding))
        # else:
        #     padding = np.zeros((X_3.shape[0] - neg_2.shape[0], neg_2.shape[1]))
        #     neg_2 = np.vstack((neg_2, padding))
        X = np.concatenate((X_3,neg_2),axis=1)
        
        # Preprocessing the data 
        # First, pad X so that it's no longer a jagged array 
        X = self.__pad_jagged_array(X) 
        
        # Splits y into three 
        # y_traj consists of the trajectory after the topk lookup 
        # Shape is (num_traj, traj_len, k)
        traj_len = X.shape[2]
        y_traj = y[:,0]
        y_traj = self.__lookup_topk(y_traj, self.topk_weights)
        y_traj = self.__pad_nan(y_traj, traj_len)
        
        # y_s_patt consists of the trajectory spatial pattern 
        # Shape is (num_traj, traj_len, 1) 
        y_s_patt = y[:,1]
        y_s_patt = self.__pad_nan(y_s_patt, traj_len)
        
        # y_s_patt consists of the trajectory temporal pattern 
        # Shape is (num_traj, traj_len, 1)  
        y_t_patt = y[:,2]
        y_t_patt = self.__pad_nan(y_t_patt, traj_len) 
        
        # Concatenate y_traj, y_s_patt, and y_t_patt
        y = np.concatenate([y_traj, y_s_patt, y_t_patt], axis = 2)
        # print(f'X shape: {X.shape}, y shape: {y.shape}')
        return X, y
        

    """
        #OLD GETITEM
        def __getitem__(self, index):
        batch_end = (index+1)*self.batch_size
        if batch_end > len(self.X):
            batch_end = len(self.X)
        X = self.X[index*self.batch_size:batch_end]
        y = self.y[index*self.batch_size:batch_end]
        
        
        # Preprocessing the data 
        # First, pad X so that it's no longer a jagged array 
        X = self.__pad_jagged_array(X) 
        
        # Splits y into three 
        # y_traj consists of the trajectory after the topk lookup 
        # Shape is (num_traj, traj_len, k)
        traj_len = X.shape[2]
        y_traj = y[:,0]
        y_traj = self.__lookup_topk(y_traj, self.topk_weights)
        y_traj = self.__pad_nan(y_traj, traj_len)
        
        # y_s_patt consists of the trajectory spatial pattern 
        # Shape is (num_traj, traj_len, 1) 
        y_s_patt = y[:,1]
        y_s_patt = self.__pad_nan(y_s_patt, traj_len)
        
        # y_s_patt consists of the trajectory temporal pattern 
        # Shape is (num_traj, traj_len, 1)  
        y_t_patt = y[:,2]
        y_t_patt = self.__pad_nan(y_t_patt, traj_len) 
        
        # Concatenate y_traj, y_s_patt, and y_t_patt
        y = np.concatenate([y_traj, y_s_patt, y_t_patt], axis = 2)
        return X, y
    """


    def __lookup_topk(self, in_array, topk_weights):
        """
        Given a numpy array consisting of all trajectories, where each 
        trajectory point is represented with a cell ID, perform a lookup to 
        get the top-k weights of the cells and return as a new numpy array 
        
        Args:
            in_array: (numpy array) Jagged array of shape 
                      (num_traj, traj_len, 1), which represents the 
                       trajectories to perform the lookup with. 
                       
        Returns:
            Array of shape (num_traj, traj_len, k) where k represents the 
            weight of each cell to its k-nearest cells 
        """
        new_array =[np.array([topk_weights[x[0]] for x in y]) for y in in_array]
        new_array = np.array(new_array,dtype = object)
        return new_array
        

    def __pad_nan(self, in_array, pad_len):
        """
        Given an array, pad every array in axis 1 (i.e. 2nd dimension) to the 
        length of the longest axis-1-array from the whole input_array. The 
        padding value is nan, the type of the elements is float and post-padding 
        is used. 
        
        Args:
            in_array: (numpy array) 4D numpy array. All the values within
                       the array must be a type in which arithmetic addition can 
                       be applied to. 
            pad_len: (integer or None) The length to pad each trajectory to. If 
                      None is provided, pad to the maximum trajectory length. 
        
        Returns:
            in_array after the padding. The padding turns a jagged array to a 
            non-jagged array, which can now be fed to the deep neural network 
            model. 
        """
        # Get some important variables from in_array shapes 
        num_data = in_array.shape[0]
        if pad_len is None:
            pad_len = max([len(x) for x in in_array])
        k = in_array[0].shape[-1]
        
        # Do the padding by creating an array of nan in the intended shape 
        # Then, we just copy the relevant values form in_array 
        final = np.empty((num_data, pad_len, k))
        final[:,:,:] = np.nan
        for i in range(len(in_array)):
            for j, row in enumerate(in_array[i]):
                final[i][j, :len(row)] = row 
        return final 


    def __pad_jagged_array(self, in_array):
        """
        Given an array, pad every array in axis 1 (i.e. 2nd dimension) to the 
        length of the longest axis-1-array from the whole input_array. The 
        padding value is 0, the type of the elements is float and post-padding 
        is used. 
        
        Args:
            in_array: (numpy array) 4D numpy array. All the values within
                       the array must be a type in which arithmetic addition can 
                       be applied to. 
        
        Returns:
            in_array after the padding. The padding turns a jagged array to a 
            non-jagged array, which can now be fed to the deep neural network 
            model. 
        """
        # Get some important variables from in_array shapes 
        num_data = in_array.shape[0]
        num_data_inner = in_array.shape[1]
        # print("num_data_inner", num_data_inner)
        max_len = max([len(y) for x in in_array for y in x])
        
        # Do the padding by creating an array of zeroes in the intended shape 
        # Then, we can perform addition to fill the relevant values in this 
        # array with the values from in_array 
        final = np.zeros((num_data,num_data_inner,max_len,1))
        
        for i in range(len(in_array)):
            for j, row in enumerate(in_array[i]):
                final[i][j, :len(row)] += row 
        return final 
        
        
class KerasPredictGenerator(keras.utils.Sequence):
    """Generator for the prediction""" 
    def __init__(self, X, batch_size, traj_len):
        self.X = X
        self.X = np.array([x[1] for x in self.X]) 
        self.X = self.__pad_jagged_array(self.X, traj_len) 
        self.X = self.X[:,:,0]
        self.batch_size = batch_size 
        
        
    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))
        
        
    def on_epoch_end(self):
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        
    
    def __getitem__(self, index):
        batch_end = (index+1)*self.batch_size
        if batch_end > len(self.X):
            batch_end = len(self.X)
            
        X = self.X[index*self.batch_size:batch_end]
        return X 
        
        
    def __pad_jagged_array(self, in_array, traj_len):
        """
        Given an array, pad every array in axis 1 (i.e. 2nd dimension) to the 
        length of the longest axis-1-array from the whole input_array. The 
        type of the elements is float and post-padding is used. 
        
        Args:
            in_array: (numpy array) 4D numpy array. All the values within
                       the array must be a type in which arithmetic addition can 
                       be applied to. 
        
        Returns:
            in_array after the padding. The padding turns a jagged array to a 
            non-jagged array, which can now be fed to the deep neural network 
            model. 
        """
        # Get important variables from in_array shapes 
        num_data = in_array.shape[0]
        
        # Do the padding by creating an array of zeroes in the intended shape 
        # Then, we can perform addition to fill the relevant values in this 
        # array with the values from in_array 
        final = np.zeros((num_data,traj_len,1))
        for j, row in enumerate(in_array):
            final[j, :len(row)] += row 
        return final 