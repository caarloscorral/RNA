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
LEARNING_RATE = 0.1
CYCLES = 20

def Adaline_Training():
    ''' weights[0] = w1       weights[1] = w2       weights[2] = w3       weights[3] = w4
        weights[4] = w5       weights[5] = w6       weights[6] = w7       weights[7] = w8 
        weights[8] = θ --> threshold                learning_rate = γ
    '''
    # Inizializing random weights and threshold
    weights = np.around(np.random.rand(9), 3)

    outputs = [[] * len(validation.values)] * CYCLES
    mse_training_list = []
    mse_validation_list = []

    # Training Adaline by the number of cycles (stop criterion)
    for _ in range(CYCLES):
        training_errors = []
        validation_errors = []

        ##### TRAINING ####
        # For each pattern get output and set new weights and threshold
        for pattern in training.values:
            # Getting output
            output = weights[0:8] @ pattern[0:8]
            output += weights[8]

            difference = pattern[8] - output

            # Setting new weights
            weights[0:8] += LEARNING_RATE * difference * pattern[0:8]
            
            # Setting new threshold
            weights[8] += LEARNING_RATE * difference

            # Appending pattern error to training errors list
            training_errors.append(difference)

        tError = np.array(training_errors)

        # For each cycle get Mean Squared Error (MSE) for training set
        mse_training = np.sum(tError ** 2)
        mse_training /= len(tError)
        mse_training_list.append(mse_training)


        ##### VALIDATION ####
        for i in range(len(validation.values)):
            #Getting output
            output = weights[0:8] @ validation.values[i][0:8]
            output += weights[8]

            difference = validation.values[i][8] - output

            # Appending pattern error to validation errors list
            validation_errors.append(difference)

            # Saving outputs into outputs table

        vError = np.array(validation_errors)

        # For each cycle get Mean Squared Error (MSE) for validation set
        mse_validation = np.sum(vError ** 2)
        mse_validation /= len(validation.values)
        mse_validation_list.append(mse_validation)

    # Printing into screen the errors table and saving into a dataframe
    error_table = list(zip(mse_training_list, mse_validation_list))
    error_table = pd.DataFrame(error_table, columns=["Error medio de entrenamiento", "Error medio de validación"])
    error_table.reset_index(inplace=True)
    error_table = error_table.rename(columns = {'index': 'Ciclo'})

    print(error_table.to_string(index=False))


    # Saving outputs for each pattern into a dataframe
    

    # Saving final weights and threshold into a dataframe
    weights_and_threshold_table = pd.DataFrame(weights)
    weights_and_threshold_table = weights_and_threshold_table.T
    weights_and_threshold_table.columns = ["Peso final w1", "Peso final w2", "Peso final w3", "Peso final w4", "Peso final w5", "Peso final w6", "Peso final w7", "Peso final w8", "Umbral final"]

    # Saving errors table, weights and threshold table, and outputs table into a Excel file
    writer = pd.ExcelWriter(r"C:\Users\carlo\OneDrive\Documentos\GitHub\RNA\Adaline.xlsx", engine = 'xlsxwriter')

    error_table.to_excel(writer, sheet_name="Errors", index=False, header=True)
    weights_and_threshold_table.to_excel(writer, sheet_name="Final weights", index=False, header=True)
    #outputs_table.to_excel(writer, sheet_name="Outputs", index=False, header=True)

    writer.save()
    writer.close()


def main():
    Adaline_Training()

    
if __name__ == "__main__" :
    main()
