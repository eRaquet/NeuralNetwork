import numpy as np
import sys
import h5py
import costFunctions
import activationFunctions
from matplotlib import pyplot as plt

class Data ():
    def __init__(self):
        pass

    def readDataFromFile(self, name):

        try:
            file = h5py.File(sys.path[0] + '\\trainingDatasets\\' + name + '.h5', 'r')

            #load input and output datasets
            self.inputSet = file['input'][:]
            self.outputSet = file['output'][:]

            file.close()
        
        except Exception as error:
            
            file.close()
            raise Exception(error)

    #seperate inputs from outputs
    def disectData(self, epochSize):

        self.trainDataIn = self.inputSet[0:epochSize]
        self.trainDataOut = self.outputSet[0:epochSize]
        self.testDataIn = self.inputSet[epochSize:]
        self.testDataOut = self.outputSet[epochSize:]

    #shuffle training data without mixing the inputs and outputs
    def shuffleTrainData(self):

        #get seed value for this shuffle
        seedVal = np.random.randint(1, 100000)

        #use the same seed to shuffle both datasets
        np.random.seed(seedVal)
        np.random.shuffle(self.trainDataIn)
        np.random.seed(seedVal)
        np.random.shuffle(self.trainDataOut)

#a layer object for the network
class Layer ():

    def __init__(self, layerSize, activation, momentumCo=0.0, nextLayer=None):

        self.layerSize = layerSize
        self.activation = activation
        self.nextLayer = nextLayer

        if self.nextLayer != None:

            self.endLayer = False
            
            self.layerWeight = np.zeros((self.layerSize, self.nextLayer.layerSize), np.float64)
            self.layerBias = np.zeros(self.nextLayer.layerSize, np.float64)
            self.weightPrime = np.zeros((self.layerSize, self.nextLayer.layerSize), np.float64)
            self.biasPrime = np.zeros(self.nextLayer.layerSize, np.float64)
            self.nextLayerPrime = None

        else:
            self.endLayer = True

            self.layerWeight = np.zeros(self.layerSize)
            self.layerBias = np.zeros(self.layerSize)
            self.weightPrime = None
            self.Prime = None

        self.layerValue = np.zeros(self.layerSize, np.float64)
        self.unactiveValue = np.zeros(self.layerSize, np.float64)

        self.momentumCo = momentumCo

    #initate the layer with random weight and bias values
    def randomWeights(self):

        #if not endLayer, give weights and biases random values between -0.25 and 0.25
        if self.endLayer == False:
            self.layerWeight = (np.random.random(self.layerWeight.shape) / 2 - 0.25)
            self.layerBias = (np.random.random(self.layerBias.shape) / 2 - 0.25)

    #assign new layer values to next layer based on weights and biases
    def activate(self):

        if self.endLayer == False:
            self.unactiveValue = self.layerValue @ self.layerWeight + self.layerBias
            self.nextLayer.layerValue = activationFunctions.activationDic[self.activation][0](self.unactiveValue)
    
    #calulate new nextLayerPrime and weight, bias partial dirivatives
    def updateGradiants(self, costPrime):

        #three cases: this layer is the end layer
        if self.endLayer == True:

            self.nextLayerPrime = np.atleast_2d(costPrime)

        #the next layer is the end layer
        elif self.nextLayer.endLayer == True:

            self.nextLayerPrime =  self.nextLayer.nextLayerPrime * activationFunctions.activationDic[self.activation][1](self.unactiveValue)

            self.weightPrime += np.atleast_2d(self.layerValue).T @ self.nextLayerPrime
            self.biasPrime += self.nextLayerPrime[0]
        
        #or we are in the middle of the hidden layers
        else:

            self.nextLayerPrime = self.nextLayer.nextLayerPrime @ self.nextLayer.layerWeight.T * activationFunctions.activationDic[self.activation][1](self.unactiveValue)
            
            self.weightPrime += np.atleast_2d(self.layerValue).T @ self.nextLayerPrime
            self.biasPrime += self.nextLayerPrime[0]

    def clearGradiants(self, momentumCo):

        #reset the layer derivatives according to the layer momentum coefficient
        if self.endLayer == False:
            self.weightPrime *= momentumCo
            self.biasPrime *= momentumCo

#neural network object
class Network ():
    def __init__(self, shape, activations, miniBatchSize, momentumCo=0):

        self.layers = []
        self.shape = shape
        self.activations = activations
        self.networkLength = len(self.shape)
        self.holdLayer = None
        self.batchCostAve = np.empty(self.shape[self.networkLength - 1])

        self.momentumCo = momentumCo

        self.miniBatchSize = miniBatchSize

        #assemble layers
        for i in range(0, len(shape)):
            self.layers.append(Layer(self.shape[self.networkLength - i - 1], activations[self.networkLength - 1 - i], self.momentumCo, self.holdLayer))
            self.layers[i].randomWeights()
            self.holdLayer = self.layers[i]
        self.layers.reverse()

        self.result = np.zeros(self.shape[self.networkLength - 1])

    #activate each layer of the network in turn to determine its output
    def activateNetwork(self, input):

        #input start value
        self.layers[0].layerValue = np.array(list(input))

        #activate each layer in succession
        for i in range(0, self.networkLength):
            self.layers[i].activate()

        #get the result at the end
        self.result = self.layers[self.networkLength - 1].layerValue
    
    #find the derivatives of the layers
    def updateAllGradiants(self, input, expectedOutput):

        #actiave the network
        self.activateNetwork(input)

        #find the cost of the network
        self.cost = costFunctions.costDic['squared'][0](self.result, expectedOutput)
        self.costDirivative = costFunctions.costDic['squared'][1](self.result, expectedOutput)

        #go through each layer, calculating dirivatives for each
        for i in range(0, self.networkLength):
            self.layers[self.networkLength - i - 1].updateGradiants(self.costDirivative)

    #move down the network gradient by some small step
    def decendGradiants(self, learnRate):

        for i in range(0, len(self.layers) - 1):
            self.layers[i].layerWeight -= self.layers[i].weightPrime * learnRate
            self.layers[i].layerBias -= self.layers[i].biasPrime * learnRate

    #clear the derivatives for each layer
    def clearAllGradiants(self):
        
        #for all layers other than endLayer
        for i in range(0, len(self.layers) - 1):
            self.layers[i].clearGradiants(self.momentumCo)

    #run the test data through the network
    def testNetwork(self, testInput, testOutput):
        
        cost = 0.0

        self.activateNetwork(testInput)
        cost = np.mean(costFunctions.costDic['squared'][0](self.result, testOutput))

        return cost

    #train the network on one minibatch
    def trainBatch(self, inputData, outputData, learnRate):

        self.inputData = inputData
        self.expectedData = outputData

        self.updateAllGradiants(self.inputData, self.expectedData)
        self.batchCostAve = np.mean(self.cost)

        #decend the network gradient
        self.decendGradiants(learnRate / self.miniBatchSize)

        self.clearAllGradiants()

    def trainNetwork(self, epochSize, data, trainIterations, learnRate, moniter=True):

        data.disectData(epochSize)

        batchCounter = 0

        #for each training iteration
        for i in range(0, trainIterations):

            #shuffle the training data
            data.shuffleTrainData()

            #for each minibatch of data
            for ii in range(0, int(epochSize / self.miniBatchSize)):
                
                inputData = data.inputSet[ii * self.miniBatchSize : (ii + 1) * self.miniBatchSize]
                outputData = data.outputSet[ii * self.miniBatchSize : (ii + 1) * self.miniBatchSize]

                self.trainBatch(inputData, outputData, learnRate)

                if moniter == True:

                    #total number of printouts
                    moniterFreq = 30

                    if batchCounter % int(epochSize / self.miniBatchSize * trainIterations / moniterFreq) == 0:

                        #generate display data
                        progString = '=' * int(moniterFreq * batchCounter / epochSize * self.miniBatchSize / trainIterations) + ' ' * int(moniterFreq - moniterFreq * batchCounter / epochSize * self.miniBatchSize / trainIterations)
                        percentage = str(batchCounter / epochSize * self.miniBatchSize / trainIterations * 100)[0:4]
                        
                        #run the test data
                        cost = str(self.testNetwork(data.testDataIn, data.testDataOut).mean())[0:7]
                        
                        print('progress: |' + progString + '| ' + percentage + '% training cost: ' + str(self.batchCostAve.mean())[0:7] + ' test cost: ' + cost, end='\r', flush = True)
                    
                batchCounter += 1

        print('\n')

#save the status of a network
def saveNetwork(network, name):

    try:
        file = h5py.File(sys.path[0] + '\\savedNetworks\\' + name + '.h5', 'a')
        
        #if file is not empty
        if len(file.values()) != 0:

            #check the compatability between the file and the network object for each layer
            for i in range(0, len(file.values())):
                if network.layers[i].layerWeight.shape != file['/layer' + str(i)]['weights'].shape or network.layers[i].layerBias.shape != file['/layer' + str(i)]['bias'].shape or network.layers[i].activation != file['/layer' + str(i)]['activation'][0].decode('UTF-8'):
                    raise Exception('Layers not compatable for this file')
                
            #write new network values over old ones
            for i in range(0, len(network.layers)):
                file['/layer' + str(i)]['weights'].write_direct(network.layers[i].layerWeight)
                file['/layer' + str(i)]['bias'].write_direct(network.layers[i].layerBias)

        #if file is empty
        else:

            #write new datasets with network values
            for i in range(0, len(network.layers)):
                file.create_group('layer' + str(i))
                file['/layer' + str(i)].create_dataset('weights', data = network.layers[i].layerWeight)
                file['/layer' + str(i)].create_dataset('bias', data = network.layers[i].layerBias)
                file['/layer' + str(i)].create_dataset('activation', data = [network.layers[i].activation])


        file.close()
    
    except Exception as error:

        file.close()
        print('Could not save network: ' + repr(error))

#load a network from a saved file
def loadNetwork(name, miniBatchSize, momentumCo=0):

    try:
        file = h5py.File(sys.path[0] + '\\savedNetworks\\' + name + '.h5', 'r')

        layers = []
        activations = []

        #load network shape and activation pattern
        for i in range(0, len(file.values())):
            layers.append(file['/layer' + str(i)]['weights'].shape[0])
            activations.append(file['/layer' + str(i)]['activation'][0].decode('UTF-8'))

        #initiate network
        network = Network(tuple(layers), tuple(activations), miniBatchSize, momentumCo)

        for i in range(0, len(network.layers)):
            file['/layer' + str(i)]['weights'].read_direct(network.layers[i].layerWeight)
            file['/layer' + str(i)]['bias'].read_direct(network.layers[i].layerBias)

        file.close()

        return network
    
    except Exception as error:
        
        try:

            file.close()

        except Exception:
            
            None

        print('Could not load network: ' + repr(error))

        return None