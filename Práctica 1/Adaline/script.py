# Importing libraries
import random
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

############################################################# PREPROCESSING #############################################################

# Getting data
data = pd.read_csv(r"Adaline\california_housing.csv")
maxx = {}
minn = {}

# Normalize data
for column in data.columns:
    maxx[column] = data[column].max()
    minn[column] = data[column].min()
    minus = (maxx[column] - minn[column])
    data[column] -= minn[column]
    data[column] /= minus

# Getting random training, testing and validation sets
training_set, testing = train_test_split(data, shuffle=True, test_size=0.4)
testing, validation = train_test_split(testing, shuffle=True, test_size=0.5)

# Reseting indexes
training_set = training_set.reset_index(drop=True)
testing = testing.reset_index(drop=True)
validation = validation.reset_index(drop=True)

# Saving training testing and validation sets into a CSV file
training_set.to_csv(r"Adaline\Outputs\Training.csv", index=False)
testing.to_csv(r"Adaline\Outputs\Testing.csv", index=False)
validation.to_csv(r"Adaline\Outputs\Validation.csv", index=False)

training_set.to_csv(r"Perceptrón multicapa\Training.csv", index=False)
testing.to_csv(r"Perceptrón multicapa\Testing.csv", index=False)
validation.to_csv(r"Perceptrón multicapa\Validation.csv", index=False)


############################################################# ADALINE ALGORITHM #############################################################

def Adaline_Training(cycles, learning_rate, training):
    ''' weights[0] = w1       weights[1] = w2       weights[2] = w3       weights[3] = w4
        weights[4] = w5       weights[5] = w6       weights[6] = w7       weights[7] = w8 
        weights[8] = θ --> threshold                learning_rate = γ
    '''
    # Inizializing random weights and threshold
    weights = np.around(np.random.rand(9), 3)

    outputs = list()
    mse_training_list = []
    mse_validation_list = []

    # Training Adaline by the number of cycles (stop criterion)
    for cycle in range(cycles):
        outputs.append(list())
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
            weights[0:8] += learning_rate * difference * pattern[0:8]

            # Setting new threshold
            weights[8] += learning_rate * difference

            # Appending pattern error to training errors list
            training_errors.append(difference)

        tError = np.array(training_errors)

        # For each cycle get Mean Squared Error (MSE) for training set
        mse_training = np.sum(tError ** 2)
        mse_training /= len(tError)
        mse_training_list.append(mse_training)

        ##### VALIDATION ####
        for i in range(len(validation.values)):
            # Getting output
            output = weights[0:8] @ validation.values[i][0:8]
            output += weights[8]

            difference = validation.values[i][8] - output

            # Appending pattern error to validation errors list
            validation_errors.append(difference)

            # Saving outputs into outputs list
            outputs[cycle].append(output)

        vError = np.array(validation_errors)

        # For each cycle get Mean Squared Error (MSE) for validation set
        mse_validation = np.sum(vError ** 2)
        mse_validation /= len(validation.values)
        mse_validation_list.append(mse_validation)

    # Printing into screen the errors table and saving into a dataframe
    error_table = list(zip(mse_training_list, mse_validation_list))
    error_table = pd.DataFrame(error_table, columns=[
                               "Error medio de entrenamiento", "Error medio de validación"])
    error_table.reset_index(inplace=True)
    error_table = error_table.rename(columns={'index': 'Ciclo'})

    print("Learning rate: " + str(learning_rate) + "\n" + error_table.to_string(index=False) +
          "\n********************************************************************************************")

    # Saving outputs into a dataframe
    outputs_table = pd.DataFrame(outputs)

    # Saving final weights and threshold into a dataframe
    weights_and_threshold_table = pd.DataFrame(weights)
    weights_and_threshold_table = weights_and_threshold_table.T
    weights_and_threshold_table.columns = ["Peso final w1", "Peso final w2", "Peso final w3",
                                           "Peso final w4", "Peso final w5", "Peso final w6", "Peso final w7", "Peso final w8", "Umbral final"]

    # Saving errors table, weights and threshold table, and outputs table into Excel adn CSV files; checking first if there are existent files and removing them
    if os.path.exists(r"Adaline\Outputs\Adaline" + str(learning_rate) + ".xlsx"):
        os.remove(r"Adaline\Outputs\Adaline" + str(learning_rate) + ".xlsx")

    elif os.path.exists(r"Adaline\Outputs\Errors" + str(learning_rate) + ".csv"):
        os.remove(r"Adaline\Outputs\Errors" + str(learning_rate) + ".csv")

    elif os.path.exists(r"Adaline\Outputs\Final weights"+str(learning_rate) + ".csv"):
        os.remove(r"Adaline\Outputs\Final weights" +
                  str(learning_rate) + ".csv")

    elif os.path.exists(r"Adaline\Outputs\Outputs" + str(learning_rate) + ".csv"):
        os.remove(r"Adaline\Outputs\Outputs" + str(learning_rate) + ".csv")

    writer = pd.ExcelWriter(r"Adaline\Outputs\Adaline" +
                            str(learning_rate) + ".xlsx", engine='xlsxwriter')

    error_table.to_excel(writer, sheet_name="Errors", index=False, header=True)
    error_table.to_csv(r"Adaline\Outputs\Errors" +
                       str(learning_rate) + ".csv", index=False)

    weights_and_threshold_table.to_excel(
        writer, sheet_name="Final weights", index=False, header=True)
    weights_and_threshold_table.to_csv(
        r"Adaline\Outputs\Weights_and_threshold" + str(learning_rate) + ".csv", index=False)

    # Denormalize outputs
    for column in outputs_table.columns:
        minus = (maxx["median_house_value"] - minn["median_house_value"])
        outputs_table[column] *= minus
        outputs_table[column] += minn["median_house_value"]
        outputs_table[column] = np.around(outputs_table[column], 2)

    outputs_table.to_excel(writer, sheet_name="Outputs", header=True)
    outputs_table.to_csv(r"Adaline\Outputs\Outputs" +
                         str(learning_rate) + ".csv", index=False)

    writer.save()
    writer.close()

    return mse_validation_list, weights


def testing_error(weights):
    testing_errors = []
    testing_outputs = []

    for pattern in testing.values:
        # Getting output
        output = weights[0:8] @ pattern[0:8]
        output += weights[8]

        difference = pattern[8] - output

        # Appending pattern error to testing errors list
        testing_errors.append(difference)

        # Saving outputs into outputs list
        testing_outputs.append(output)

    # For each cycle get Mean Squared Error (MSE) for testing set
    testing_errors = np.array(testing_errors)
    mse_testing = np.sum(testing_errors ** 2)
    mse_testing /= len(testing_errors)

    # Denormalize outputs
    for index, output in enumerate(testing_outputs):
        minus = (maxx["median_house_value"] - minn["median_house_value"])
        output *= minus
        output += minn["median_house_value"]
        testing_outputs[index] = np.around(output, 2)

    # Saving final weights and threshold into a dataframe
    weights = np.append(weights, mse_testing)
    weights_and_threshold_table = pd.DataFrame(weights)
    weights_and_threshold_table = weights_and_threshold_table.T
    weights_and_threshold_table.columns = ["Peso final w1", "Peso final w2", "Peso final w3",
                                           "Peso final w4", "Peso final w5", "Peso final w6", "Peso final w7", "Peso final w8", "Umbral final", "MSE Testing Final"]
    
    # Saving final outputs into a dataframe
    testing_outputs_table = pd.DataFrame(testing_outputs)

    return weights_and_threshold_table, testing_outputs_table


def main():
    MAX_CYCLES = 100
    errors = []
    cycles = []
    weight_list = []
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1]

    for rate in learning_rates:
        error, weights = Adaline_Training(MAX_CYCLES, rate, training_set)
        newCycles = error.index(min(error)) + 1
        cycles.append(newCycles)
        bestError, weights = Adaline_Training(newCycles, rate, training_set)
        errors.append(min(bestError))
        weight_list.append(weights)

    best_learning_rate_index = errors.index(min(errors))
    best_learning_rate = learning_rates[best_learning_rate_index]
    weights_and_threshold_table, testing_outputs_table = testing_error(weight_list[best_learning_rate_index])

    # Saving errors testing table, weights and threshold table, and outputs table into Excel adn CSV files; checking first if there are existent files and removing them
    if os.path.exists(r"Adaline\Outputs\FinalWeights.csv"):
        os.remove(r"Adaline\Outputs\FinalWeights.csv")

    elif os.path.exists(r"Adaline\Outputs\FinalOutputs.csv"):
        os.remove(r"Adaline\Outputs\FinalOutputs.csv")

    writer = pd.ExcelWriter(r"Adaline\Outputs\FinalWeights.xlsx", engine='xlsxwriter')
    weights_and_threshold_table.to_excel(writer, index=False, header=True)
    weights_and_threshold_table.to_csv(r"Adaline\Outputs\FinalWeights.csv", index=False)

    writer = pd.ExcelWriter(r"Adaline\Outputs\FinalOutputs.xlsx", engine='xlsxwriter')
    testing_outputs_table.to_excel(writer, index=False, header=True)
    testing_outputs_table.to_csv(r"Adaline\Outputs\FinalOutputs.csv", index=False)

    writer.save()
    writer.close()

    if os.path.exists(r"Adaline\Outputs\Errors.csv"):
        os.remove(r"Adaline\Outputs\Errors.csv")

    error_table = list(zip(learning_rates, cycles, errors))
    error_table = pd.DataFrame(error_table, columns=[
                               "Ratio de aprendizaje", "Ciclos óptimos", "Error de validación"])
    error_table.reset_index(drop=True)
    error_table.to_csv(r"Adaline\Outputs\Errors.csv", index=False)


if __name__ == "__main__":
    main()
