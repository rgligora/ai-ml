import numpy
from collections import OrderedDict

class NeuralNetwork():

    def __init__(self, trainingData, testData, neuralNetworkArch):
        self.trainingData = trainingData
        self.testData = testData
        self.trainingDataExpectedOutputs = []
        self.trainingDataOutputs = []
        self.neuralNetworkArch = neuralNetworkArch
        self.wightMatricies = []
        self.biasVectors = []


    def sigmoidActivationFunction(self, x):
        return 1 / (1 + numpy.exp(-x))
    
    def meanSquaredError(self):
        N = len(self.trainingDataExpectedOutputs)
        sum = 0
        for i in range(N):
            sum += (self.trainingDataExpectedOutputs[i] - self.trainingDataOutputs[i]) * (self.trainingDataExpectedOutputs[i] - self.trainingDataOutputs[i])
        return sum/N
    
    
    def build(self):
        prevLayerNeurons = self.neuralNetworkArch[0]
        for layerNeurons in self.neuralNetworkArch[1:]:
            W = numpy.random.normal(0, 0.01, (int(layerNeurons), int(prevLayerNeurons)))
            B = numpy.random.normal(0, 0.01, (int(layerNeurons)))
            self.wightMatricies.append(W)
            self.biasVectors.append(B)
            prevLayerNeurons = layerNeurons
        
        
    def run(self, row):
        layerDepth = 0
        inputLayer0 = list(row.values())
        expectedOutput = inputLayer0.pop()
        self.trainingDataExpectedOutputs.append(float(expectedOutput))
        inputLayer0 = [float(value) for value in inputLayer0]
        outputs = []
        outputs.append(numpy.array(inputLayer0))
        
        for layerNeurons in self.neuralNetworkArch[1:]:
            layerDepth = layerDepth + 1
            layerInput = outputs[layerDepth-1]
            layerOutput = self.wightMatricies[layerDepth-1].dot(layerInput) + self.biasVectors[layerDepth-1]
            if layerDepth < len(self.neuralNetworkArch)-1:
                sigmoidActivationFunctionOutput = numpy.empty_like(layerOutput)
                for i in range(layerOutput.size):
                    sigmoidActivationFunctionOutput[i] = self.sigmoidActivationFunction(layerOutput[i])
                outputs.append(sigmoidActivationFunctionOutput)
            else:
                outputs.append(layerOutput)
        self.trainingDataOutputs.append(outputs[layerDepth])
        
        
    
                 
class GeneticAlgorithm():
    
    def __init__(self, trainingData, testData, neuralNetworkArch, populationSize, elitism, mutationProbability, stdDeviation, iterations):
        self.trainingData = trainingData
        self.testData = testData
        self.neuralNetworkArch = self.parseNeuralNetworkArch(neuralNetworkArch, trainingData)
        self.populationSize = populationSize
        self.elitism = elitism
        self.mutationProbability =mutationProbability
        self.stdDeviation = stdDeviation
        self.iterations = iterations
        self.bestUnit = None
    
        
    #This function parses the architecture of the NN
    #First element in the structure is the number of neurons in the input layer 
    #Last element in the structure is the number of neurons in the output layer
    def parseNeuralNetworkArch(self, neuralNetworkArch, trainingData):
        neuronsPerLayer = []
        neuronsPerLayer.append(str(len(trainingData[0]) - 1))
        split = neuralNetworkArch.split('s')
        while '' in split:
            split.remove('')
        neuronsPerLayer = neuronsPerLayer + split
        neuronsPerLayer.append('1')
        return neuronsPerLayer
        
    
    def train(self):
        population = {}
        for unit in range(self.populationSize):
            nn = NeuralNetwork(self.trainingData, self.testData, self.neuralNetworkArch)
            nn.build()
            for row in self.trainingData:
                nn.run(row)
            error = nn.meanSquaredError()/2
            population[float(error)] = [nn.wightMatricies, nn.biasVectors]
        population = OrderedDict(sorted(population.items()))
        
        for iteration in range(self.iterations):
            newPopulation = {}
            for i, (error, values) in enumerate(population.items()):
                if i < self.elitism:
                    newPopulation[error] = values
                else:
                    break
            newPopulation = OrderedDict(sorted(newPopulation.items()))
            populationErrors = list(population.keys())
            minError = min(populationErrors)
            maxError = max(populationErrors)
            fitness = [(maxError - error) for error in populationErrors]
            
            total_fitness = sum(fitness)
            probability = [fit / total_fitness for fit in fitness]
            
            while len(newPopulation) < self.populationSize:
                parent1Key = numpy.random.choice(list(populationErrors), p=probability)
                parent2Key = numpy.random.choice(list(populationErrors), p=probability)
                while parent1Key == parent2Key:
                    parent2Key = numpy.random.choice(list(populationErrors), p=probability)
                
                parent1 = population[parent1Key]
                parent2 = population[parent2Key]
                
                #Crossing weights
                childWeightMatricies = []
                for i in range(len(parent1[0])):
                    weight1 = parent1[0][i]
                    weight2 = parent2[0][i]
                    averageWeights = []
                    for j in range(len(weight1)):
                        averageRow = []
                        for k in range(len(weight1[j])):
                            averageRow.append((weight1[j][k] + weight2[j][k]) / 2)
                        averageWeights.append(numpy.array(averageRow))
                    childWeightMatricies.append(numpy.array(averageWeights))
                childWeightMatricies = [numpy.array(matrix) for matrix in childWeightMatricies]
                
                #Crossing bias
                childBiasVectors = []
                for i in range(len(parent1[1])):
                    bias1 = parent1[1][i]
                    bias2 = parent2[1][i]
                    averageBiases = []
                    for j in range(len(bias1)):
                        averageBiases.append((bias1[j] + bias2[j]) / 2)
                    childBiasVectors.append(numpy.array(averageBiases))
                childBiasVectors = [numpy.array(vector) for vector in childBiasVectors]
                
                #Mutation
                for chromosomeWeights in range(len(childWeightMatricies)):
                    if numpy.random.rand() <= self.mutationProbability:
                        childWeightMatricies[chromosomeWeights] += numpy.random.normal(0, self.stdDeviation, childWeightMatricies[chromosomeWeights].shape)
                for chromosomeBias in range(len(childBiasVectors)):
                    if numpy.random.rand() <= self.mutationProbability:
                        childBiasVectors[chromosomeBias] += numpy.random.normal(0, self.stdDeviation, childBiasVectors[chromosomeBias].shape)
                
                
                childNn = NeuralNetwork(self.trainingData, self.testData, self.neuralNetworkArch)
                childNn.wightMatricies = childWeightMatricies
                childNn.biasVectors = childBiasVectors
                for row in self.trainingData:
                    childNn.run(row)
                childError = childNn.meanSquaredError()/2
                newPopulation[float(childError)] = [childNn.wightMatricies, childNn.biasVectors]
                
            population = OrderedDict(sorted(newPopulation.items()))
            
            bestUnitKey = list(population.keys())[0]
            bestUnit = NeuralNetwork(self.trainingData, self.testData, self.neuralNetworkArch)
            bestUnit.wightMatricies = population[bestUnitKey][0]
            bestUnit.biasVectors = population[bestUnitKey][1]
            for row in self.trainingData:
                bestUnit.run(row)
            bestUnitMSE = bestUnit.meanSquaredError()
            if iteration % 2000 == 0 and iteration != 0:
                print(f"[Train error @{iteration}]: {float(bestUnitMSE):.6f}")
                
        print(f"[Train error @{self.iterations}]: {float(bestUnitMSE):.6f}")
        self.bestUnit = bestUnit
            
    def test(self):
        for row in self.testData:
                self.bestUnit.run(row)
        print(f"[Test error]: {float(self.bestUnit.meanSquaredError()):.6f}")
        