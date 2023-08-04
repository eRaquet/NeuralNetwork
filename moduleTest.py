import neuralNetwork as Net
from matplotlib import pyplot as plt
import numpy as np

#you can load a saved network as follows
net = Net.loadNetwork('exampleNetwork', 200)

#using numpy, you can visualize the output of different end nodes in the network
a = np.ndarray((100, 100))
for i in range(0, 100):
    for j in range(0, 100):

        #activate the network with the desired inputs
        net.activateNetwork((i / 100, j / 100))
        
        #read the result of the first end node
        a[i][j] = net.result[0]

plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()

#you can also create a new network of custom size and activation
net2 = Net.Network((2, 20, 20, 15, 10, 2), ('GeLu', 'GeLu', 'GeLu', 'GeLu', 'GeLu', 'Linear'), 100, 0.4)

a = np.ndarray((100, 100))
for i in range(0, 100):
    for j in range(0, 100):

        net2.activateNetwork((i / 100, j / 100))
        a[i][j] = net2.result[0]

plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()

#you can load data from a saved dataset as follows
data = Net.Data()
data.readDataFromFile('exampleDataset')
print('The loaded dataset contians ' + str(len(data.inputSet)) + ' datum')

#you can disect the data into test and train data
data.disectData(15000) #take 15000 datasets to train with--leave the rest for testing
print('The network has ' + str(len(data.trainDataIn)) + ' training datasets')

#you can train the network on the data...
net2.trainNetwork(15000, data, 100, 0.3)

#...and see the outcome!
a = np.ndarray((100, 100))
for i in range(0, 100):
    for j in range(0, 100):

        net2.activateNetwork((i / 100, j / 100))
        a[i][j] = net2.result[0]

plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()

#you can also save the network to load it in again later
Net.saveNetwork(net2, 'exampleNetwork_2')