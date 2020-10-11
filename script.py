# Importing libraries
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

################################## PREPROCESSING ###################################

# Getting data
data = pd.read_csv("california_housing.csv")

# Normalize data
for column in data.columns:
    maxx = data[column].max()
    minn = data[column].min()
    minus = (maxx-minn)
    data[column] -= minn
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

def Adaline_Training(cycles):
    ''' weights[0] = w1       weights[1] = w2       weights[2] = w3       weights[3] = w4
        weights[4] = w5       weights[5] = w6       weights[6] = w7       weights[7] = w8 
        learning_rate = γ                           threshold = θ
        '''
    # Inizializing random weights and threshold
    weights = np.around(np.random.rand(1, 8), 3)

    threshold = random.random() / 1000.0

    mse_training_list = list()
    mse_validation_list = list()

    # Training Adaline by the number of cycles (stop criterion)
    for _ in range(cycles):
        training_errors = list()
        validation_errors = list()

        ##### TRAINING ####
        # For each pattern get output and set new weights and threshold
        for pattern in training.values:
            # Getting output
            output = weights @ pattern[0:8]
            # output += threshold

            difference = pattern[8] - output

            # Setting new weights
            weights += learning_rate * difference * pattern[0:8]
            
            # Setting new threshold
            threshold += learning_rate * difference

            # Appending pattern error to training errors list
            training_errors.append(difference)

        tError = np.array(training_errors)

        # For each cycle get Mean Squared Error (MSE) for training set
        mse_training = np.sum(tError ** 2)
        mse_training /= len(tError)
        mse_training_list.append(mse_training)


        ##### VALIDATION ####
        for pattern in validation.values:
            #Getting output
            output = weights @ pattern[0:8]
            # output += threshold

            difference = pattern[8] - output

            # Appending pattern error to validation errors list
            validation_errors.append(difference)

        vError = np.array(validation_errors)

        # For each cycle get Mean Squared Error (MSE) for validation set
        mse_validation = np.sum(vError ** 2)
        mse_validation /= len(validation.values)
        mse_validation_list.append(mse_validation)

        if mse_training < threshold:
            #break
            pass
    
    # Printing into screen
    print(mse_training_list)


def main():
    Adaline_Training(20)

    
if __name__ == "__main__" :
    main()
