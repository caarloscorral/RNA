# Importing libraries
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

############################################################# PREPROCESSING #############################################################

# Getting data
data = pd.read_csv("california_housing.csv")

# Normalize data
for column in data.columns:
    max = data[column].max()
    min = data[column].min()
    minus = (max-min)
    data[column] -= min
    data[column] /= minus

# Getting random training, testing and validation sets
training, testing = train_test_split(data, shuffle=True, test_size=0.4)
testing, validation = train_test_split(testing, shuffle=True, test_size=0.5)

# Reseting indexes
training = training.reset_index(drop=True)
testing = testing.reset_index(drop=True)
validation = validation.reset_index(drop=True)

#print(training)
#print(testing)
#print(validation)


############################################################# ADALINE ALGORITHM #############################################################

##### Variables ####
learning_rate = 0.3
threshold = 0
weights = list()

def Adaline_Training(cycles):
    ''' weights[0] = w1       weights[1] = w2       weights[2] = w3       weights[3] = w4
        weights[4] = w5       weights[5] = w6       weights[6] = w7       weights[7] = w8 
        learning_rate = γ                           threshold = θ
        '''
    # Inizializing random weights and threshold
    for i in range(8):
        weights.append(round(random.random(), 3))

    threshold = round(random.random(), 3)

    mse_training_list = list()
    mse_validation_list = list()

    # Training Adaline by the number of cycles (stop criterion)
    for cycle in range(1, cycles + 1):
        training_errors = list()
        validation_errors = list()

        ##### TRAINING ####
        # For each pattern get output and set new weights and threshold
        for pattern in training.values:
            # Getting output
            output = 0
            for x in range(len(pattern) - 1):
                output += weights[x] * pattern[x]
            output += threshold

            difference = pattern[8] - output

            # Setting new weights
            for x in range(len(pattern)):
                if x != 8:
                    weights[x] += learning_rate * difference * pattern[x]
                else:
                    threshold += learning_rate * difference
            
            # Setting new threshold
            threshold += learning_rate * difference

            # Appending pattern error to training errors list
            training_errors.append(difference)

        # For each cycle get Mean Squared Error (MSE) for training set
        mse_training = 0
        for error in training_errors:
            mse_training += error ** 2
        mse_training /= len(training.values)
        mse_training_list.append(mse_training)


        ##### VALIDATION ####
        for pattern in validation.values:
           # Getting output
            output = 0
            for x in range(len(pattern) - 1):
                output += weights[x] * pattern[x]
            output += threshold

            difference = pattern[8] - output

            # Appending pattern error to validation errors list
            validation_errors.append(difference)

        # For each cycle get Mean Squared Error (MSE) for validation set
        mse_validation = 0
        for error in validation_errors:
            mse_validation += error ** 2
        mse_validation /= len(validation.values)
        mse_validation_list.append(mse_validation)
        
    # Printing into screen


def main():
    Adaline_Training(20)

    
if __name__ == "__main__" :
    main()