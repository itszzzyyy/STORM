"""Reads the trajectories from the input trajectory file"""
from datetime import datetime
from os import listdir
from os.path import isfile, join
from shapely.geometry import Point
from shapely.geometry import Polygon 
import ast 
import numpy as np 

import pathlib 

class FileReader():
    """
    This class handles the processing of the trajectories. It reads the 
    trajectories from the .csv files and outputs the data in the format 
    ready to be used in the training model
    """
    def __init__(self):
        """Simply initialize important constants"""
        self.__MINUTES_IN_DAY = 1440 
        self.__SECONDS_IN_DAY = 86400
        self.__PORTO_SECOND_INCREMENT = 15 
        self.__SECONDS_IN_MINUTE = 60 
        self.__DEFAULT_TRAIN_SIZE_FRACTION = 0.7
        self.__DEFAULT_VALIDATION_SIZE_FRACTION = 0.2
        self.__DEFAULT_TEST_SIZE_FRACTION = 0.1
        
            
    def read_trajectory_from_file(self, in_path, dataset_mode, 
                                  min_trajectory_length, max_trajectory_length,
                                  bbox_coords, traj_nums): 
        """
        Reads the input_file while doing some pruning which are: removing 
        trajectories that are too short, removing trajectories that are too, 
        long, and removing trajectory points outside the valid area. The way 
        the data is read depends on the data_mode 
        
        Args:
            in_path: (String) The path to the input  
            data_mode: (String) The mode used to read the data. Different 
                       dataset require different ways of reading and 
                       consequently, different data_mode 
            min_trajectory_length: (Integer) The shortest allowable trajectory 
                                   length 
            max_trajectory_length: (Integer) The longest allowable trajectory 
                                   length 
            bbox_coords: (list of floats) Min lat, min lng, max lat and max lng
                         that represents the valid area. Points outside this 
                         area are to be removed 
            traj_nums: (list of integers) For the Porto data, this is a list 
                        containing the number of lines in the .csv trajectory 
                        to be assigned to the training, and validation data 
                        accordingly. For the Didi data, this is a list 
                        containing the number of trajectories to be assigned 
                        to the training and validation. 
            
        Returns:    
            A list of trajectories. Each trajectory is a list consisting of 
            latitude, longitude and timestamp in the form of minutes-in-day
        """
        [min_lat, min_lng, max_lat, max_lng] = bbox_coords
        self.bbox = Polygon([(min_lat, min_lng), (max_lat, min_lng), 
                             (max_lat, max_lng), (min_lat, max_lng),
                             (min_lat, min_lng)])
        
        # Read the .csv file line-by-line and process it according to the 
        # data_mode 
        in_file = open(in_path, 'r')
        if dataset_mode == 'porto':
            return(self.__read_porto(in_file, min_trajectory_length, 
                                     max_trajectory_length, traj_nums))
        elif dataset_mode == 'didi':
            return(self.__read_didi(in_file, min_trajectory_length, 
                                     max_trajectory_length, traj_nums))
        elif dataset_mode == 'ais':
            return (self.__read_AIS(in_file, min_trajectory_length,
                                     max_trajectory_length, traj_nums))
        elif dataset_mode == 'test':
            return (self.__read_AIS(in_file, min_trajectory_length,
                                    max_trajectory_length, traj_nums))
        else:
            raise ValueError("'" + dataset_mode + "' not supported.")

        in_file.close()
    

    def read_npy(self, input_directory, file_name):
        """
        A general purpose function to read .npy files. If all you need to do is 
        to read an .npy file from somewhere and don't need to do any form of 
        preprocessing, use this. 
        
        Args:
            input_directory: (string) The directory where the file is located 
            file_name: (string) The file name 
            
        Returns:
            A numpy array containing the contents of the .npy file 
        """
        fullpath = pathlib.Path(input_directory) / (file_name + ".npy")
        return np.load(fullpath, allow_pickle = True)


    def __read_AIS(self, in_file, min_trajectory_length,
                     max_trajectory_length, traj_nums):
        in_file.readline()
        [num_train, num_validation] = traj_nums
        all_train = []
        all_validation = []

        # Need to keep track of the actual number of lines read
        num_lines = 0

        for line in in_file:
            # if num_lines==1:
            #     break
            num_lines += 1
            reads = ast.literal_eval(line.strip(' ').strip('()').split('", "')[-1])

            trajectory = eval(reads[1])
            # print(trajectory)

            # Only process the trajectory further if it's not too long or too
            # short
            #print("轨迹长度",len(trajectory))
            # print("轨迹",trajectory)
            if (len(trajectory) <= max_trajectory_length and
                    len(trajectory) >= min_trajectory_length):

                # Process the trajectory by checking coordinates
                new_traj = self.__check_point(trajectory)
                #print("new_traj",new_traj)
                # The new trajectory may be shorter because points outside of
                # the area are removed. If it is now shorter, we ignore it
                if (len(new_traj) >= min_trajectory_length):
                    # Add to either the training, or validation list
                    if num_lines <= num_train:
                        all_train.append(new_traj)
                        print("READING TRAINING DATA %d" % (num_lines))
                    elif num_lines < num_validation + num_train:
                        all_validation.append(new_traj)
                        print("READING VALIDATION DATA %d" % (num_lines))
                    else:
                        break
        return [all_train, all_validation]
    # def __read_AIS(self, in_file, min_trajectory_length,
    #                  max_trajectory_length, traj_nums):
    #     in_file.readline()
    #     [num_train, num_validation] = traj_nums
    #     all_train = []
    #     all_validation = []
    #
    #     # Need to keep track of the actual number of lines read
    #     num_lines = 0
    #
    #     for line in in_file:
    #         # if num_lines==1:
    #         #     break
    #         num_lines += 1
    #         reads = ast.literal_eval(line.strip(' ').strip('()').split('", "')[-1])
    #
    #         trajectory = eval(reads[2])
    #         # print(trajectory)
    #
    #         # Only process the trajectory further if it's not too long or too
    #         # short
    #         #print("轨迹长度",len(trajectory))
    #         # print("轨迹",trajectory)
    #         if (len(trajectory) <= max_trajectory_length and
    #                 len(trajectory) >= min_trajectory_length):
    #             start_dtime = datetime.fromtimestamp(int(reads[1]))
    #
    #             start_second = (start_dtime.hour * 3600 +
    #                             start_dtime.minute * 60 + start_dtime.second)
    #
    #             # Process the trajectory by checking coordinates
    #             new_traj = self.__check_point_and_add_timestamp(trajectory,
    #                                                             start_second)
    #             #print("new_traj",new_traj)
    #             # The new trajectory may be shorter because points outside of
    #             # the area are removed. If it is now shorter, we ignore it
    #             if (len(new_traj) >= min_trajectory_length):
    #                 # Add to either the training, or validation list
    #                 if num_lines <= num_train:
    #                     all_train.append(new_traj)
    #                     print("READING TRAINING DATA %d" % (num_lines))
    #                 elif num_lines < num_validation + num_train:
    #                     all_validation.append(new_traj)
    #                     print("READING VALIDATION DATA %d" % (num_lines))
    #                 else:
    #                     break
    #     return [all_train, all_validation]

    def __read_porto(self, in_file, min_trajectory_length,
                     max_trajectory_length, traj_nums):
        """
        Reads the porto trajectory file line-by-line. Also keep track of the 
        actual number of lines read 
        
        Args:
            in_file: (file) The input porto trajectory file 
            min_trajectory_length: (Integer) The shortest allowable trajectory 
                                   length 
            max_trajectory_length: (Integer) The longest allowable trajectory 
                                   length 
            traj_nums: (list of integers) A list containing the number of 
                        trajectories for each training and validation data
        Returns:    
            A list of trajectories, and the actual number of lines read. Each 
            trajectory is a list consisting of latitude, longitude and timestamp 
            in the form of minutes-in-day
        """
        # Throws away the .csv header and then read line-by-line 
        in_file.readline()
        
        # Get the lines into the training, and validation lists
        [num_train, num_validation] = traj_nums
        all_train = []
        all_validation = []
        
        # Need to keep track of the actual number of lines read 
        num_lines = 0
        
        for line in in_file:
            num_lines += 1
            trajectory = ast.literal_eval(line.split('","')[-1].replace('"',''))
            # Only process the trajectory further if it's not too long or too 
            # short 
            if (len(trajectory) <= max_trajectory_length and 
                len(trajectory) >= min_trajectory_length):

                # Convert raw timestamp (seconds from epoch) to datetime 
                # and then convert to seconds-in-day
                start_dtime = datetime.fromtimestamp(int(line.split('","')[5]))

                start_second = (start_dtime.hour * 3600 + 
                                start_dtime.minute * 60 + start_dtime.second)
                print("时间：")
                print(start_second)
                # Process the trajectory by checking coordinates and adding 
                # timestamp 
                new_traj = self.__check_point_and_add_timestamp(trajectory, 
                                                                start_second)
                #print(new_traj)
                # The new trajectory may be shorter because points outside of 
                # the area are removed. If it is now shorter, we ignore it
                if (len(new_traj) >= min_trajectory_length):
                    # Add to either the training, or validation list
                    if num_lines <= num_train:
                        all_train.append(new_traj)
                        print("READING TRAINING DATA %d" % (num_lines))
                    elif num_lines < num_validation + num_train:
                        all_validation.append(new_traj)
                        print("READING VALIDATION DATA %d" % (num_lines))
                    else:
                        break
            return [all_train, all_validation]
        

    def __read_didi(self, in_file, min_trajectory_length, 
                     max_trajectory_length, traj_nums):
        """
        Reads the didi trajectory file line-by-line. Also keep track of the 
        actual number of lines read 
        
        Args:
            in_file: (file) The input didi trajectory file 
            min_trajectory_length: (Integer) The shortest allowable trajectory 
                                   length 
            max_trajectory_length: (Integer) The longest allowable trajectory 
                                   length 
            traj_nums: (list of integers) A list containing the number of 
                        trajectories for each training and validation data
        Returns:    
            A list of trajectories, and the actual number of lines read. Each 
            trajectory is a list consisting of latitude, longitude and timestamp 
            in the form of minutes-in-day
        """
        # Throws away the .csv header and then read line-by-line 
        in_file.readline()
        
        # Get the lines into the training and validation lists
        [num_train, num_validation] = traj_nums
        all_train = []
        all_validation = []
        
        # Need to keep track of the actual number of lines read 
        num_lines = 0
        
        for line in in_file:
            num_lines += 1
            trajectory = ast.literal_eval(line.split('","')[-1].replace('"',''))
            
            # Only process the trajectory further if it's not too long or too 
            # short 
            if (len(trajectory) <= max_trajectory_length and 
                len(trajectory) >= min_trajectory_length):
                
                # Process the trajectory by checking coordinates 
                new_traj = self.__check_point(trajectory)
                
                # The new trajectory may be shorter because points outside of 
                # the area are removed. If it is now shorter, we ignore it 
                if (len(new_traj) >= min_trajectory_length):
                    # Add to either the training, or validation list 
                    if num_lines <= num_train:
                        all_train.append(new_traj)
                        print("READING TRAINING DATA %d" % (num_lines))
                    elif num_lines < num_validation + num_train:
                        all_validation.append(new_traj)
                        print("READING VALIDATION DATA %d" % (num_lines))
                    else:
                        break 
        return [all_train, all_validation]
        

    def __check_point(self, trajectory):
        """
        Given a trajectory consisting of latitude, longitude and timestamp, 
        check if each point is inside the valid area. If it is not, remove it.
        
        Args:
            trajectory: (list) List of list of longitude, latitude and timestamp
                          
        Returns:
            A list of list of latitude, longitude and timestamp in the form 
            of minutes-in-day
        """
        new_trajectory = []
        for point in trajectory:
            shapely_point = Point(point[1], point[0])
            if self.bbox.contains(shapely_point):
                new_trajectory.append([point[1], point[0], point[2]])
        return new_trajectory
            
        
        
    def __check_point_and_add_timestamp(self, trajectory, start_second):
        """
        Given a trajectory consisting of latitude and longitude points, check if 
        each point is inside the valid area. If it is not, remove it, if it is,
        add the minutes-in-day timestamp. We also flip the ordering between 
        lat and lng, because the raw Porto data has the longitude first. 
        
        Args:
            trajectory: (list) List of list of longitude and latitude points 
            start_second: (integer) The second-in-the-day where the trajectory 
                          starts
                          
        Returns:
            A list of list of latitude, longitude and timestamp in the form 
            of minutes-in-day
        """
        # We add the minutes in day information, but for the Porto dataset, 
        # each trajectory point is 15 seconds apart, so we need both the 
        # second and minute information 
        cur_second = start_second
        new_trajectory = []
        for point in trajectory:
            # After the 15 seconds addition, cur_second may pass the max.
            # number of seconds in a day. We fix this.
            if cur_second >= self.__SECONDS_IN_DAY:
                cur_second -= self.__SECONDS_IN_DAY

            # Check if the point is inside the bbox. If it is, add time info and
            # append the point to new_trajectory
            shapely_point = Point(point[1], point[0])
            if self.bbox.contains(shapely_point):
                cur_minute = int(cur_second / self.__SECONDS_IN_MINUTE)
                new_trajectory.append([point[1], point[0], cur_minute])

            # Add 15 seconds for the next trajectory point
            cur_second += self.__PORTO_SECOND_INCREMENT
        return new_trajectory