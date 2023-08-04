import numpy as np
from matplotlib import pyplot as plt
import h5py
import sys

inputSize = 2
outputSize = 2

noise = 0.1

def dataFunction(input):
    x, y = input[0], input[1]
    return np.array([np.e**x + np.sin(16*y), np.cos(x * y * 2)])

a = np.ndarray((100, 100))
for i in range(0, 100):
    for j in range(0, 100):
        a[i][j] = (dataFunction((i / 100, j / 100)) + np.random.random(outputSize) * noise)[1]
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()

#number of datasets
datasetNum = 16000

print('Enter name for dataset: ', end='')

name = input()

print('')

try:
    file = h5py.File(sys.path[0] + '\\trainingDatasets\\' + name + '.h5', 'a')
    
    #if file is empty (and has to be formated)
    if len(file.values()) == 0:

        inputData = np.zeros((datasetNum, inputSize))
        outputData = np.zeros((datasetNum, outputSize))
            
        #format file
        for i in range(0, datasetNum):

            #create data
            inputData[i] = np.random.random(inputSize)
            outputData[i] = dataFunction(inputData[i]) + np.random.random(outputSize) * noise

        file.create_dataset('input', data=inputData)
        file.create_dataset('output', data=outputData)

    #file is not empty (and has to be overwritten)
    else:

        inputData = np.zeros((datasetNum, inputSize))
        outputData = np.zeros((datasetNum, outputSize))
            
        #format file
        for i in range(0, datasetNum):

            #create data
            inputData[i] = np.random.random(inputSize)
            outputData[i] = dataFunction(inputData[i])

        file['input'].write_direct(inputData)
        file['output'].write_direct(outputData)

    file.close()

except Exception as error:

    file.close()
    print('Could not create data: ' + repr(error))