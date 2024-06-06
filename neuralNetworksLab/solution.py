import csv
import argparse
from neuralnet import NeuralNetwork, GeneticAlgorithm
import numpy

#inicijalizacija parsera za cli argumente
def initParser():
    #https://docs.python.org/3/library/argparse.html#module-argparse
    #https://docs.python.org/3/howto/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Training data')
    parser.add_argument('--test', type=str, help='Testing data')
    parser.add_argument('--nn', type=str, help='Neural Network Architecture')
    parser.add_argument('--popsize', type=int, help='Population Size')
    parser.add_argument('--elitism', type=int, help='Elitism')
    parser.add_argument('--p', type=float, help='Mutation Probability')
    parser.add_argument('--K', type=float, help='Standard Deviation of Gauss Noise')
    parser.add_argument('--iter', type=int, help='Iterations') 
    return parser

#ucitavanje csv datoteke
def load(filePath):
    #https://docs.python.org/3/library/csv.html dokumentacija
    with open(filePath, 'r') as csvFile:
        reader = csv.DictReader(csvFile)
        data = [row for row in reader]
    return data


parser = initParser()
args = parser.parse_args()

trainingData = load(args.train)
testData = load(args.test)

geneticAlg = GeneticAlgorithm(trainingData, testData, args.nn, args.popsize, args.elitism, args.p, args.K, args.iter)

geneticAlg.train()
geneticAlg.test()
