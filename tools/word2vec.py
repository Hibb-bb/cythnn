import os, math, re
import numpy as np
import struct

from numpy import int32, uint64, float32

from tools.matrix import createMatrices, normalize as normalize_matrix
from tools.worddict import Vocabulary, Word, normalize

# creates a solution space for a word2vec model, 2 weight matrices, initialized resp. with random and with zeros
def createW2V(model, input_size, output_size):
    #print("createW2V", input_size, model.vectorsize, output_size)
    matrices = createMatrices([input_size, model.vectorsize, output_size], [2, 0])
    model.setSolution(matrices)

# returns the embedding for a Word (to be looked up in model.vocab)
def getVector(model, word):
    return model.matrices[0][word.index]

# saves the embeddings from a trained solution in a model to file
def save(fname, model, binary=False, normalize=False):
    s = sorted(model.vocab.items(), key=lambda x: x[1].index)
    solution = model.getSolution()
    if normalize:
        print("normalizing")
        normalize_matrix(model.matrices[0])
    if binary:
        with open(fname, 'wb') as fout:
            fout.write(("%s %s\n" % (len(model.vocab), solution.getLayerSize(1))).encode())
            for word, obj in s:
                row = getVector(model, obj)
                fout.write((word + " ").encode())
                fout.write(struct.pack('%sf' % len(row), *row))
    else:
        with open(fname, 'w') as fout:
            fout.write("%s %s\n" % (len(model.vocab), solution.getLayerSize(1)))
            for word, obj in s:
                row = getVector(model, obj)
                fout.write("%s %d %s\n" % (word, obj.count, ' '.join("%f" % val for val in row)))

# loads the embeddings saved to file
def load(fname, binary=False, normalized=False):
    with open(fname, 'r') as fin:
        header = fin.readline()
        wordcount, vector_size = [ int(x) for x in header.split(" ") ]
        solution = createW2V(wordcount, vector_size)
        vocab = Vocabulary({}, MIN_TF=-1)
        index = 0
        for line in fin.readlines():
            terms = line.split(" ")
            word = terms[0]
            count = int(terms[1])
            solution.matrix[0][index] = [float32(terms[i]) for i in range(2, len(terms))]
            vocab[word] = Word(count, index=index, word=word)
            index+=1
            vocab.total_words += count
        if normalized:
            for i in range(wordcount):
                solution[0][i] = normalize(solution[0][i])
        return vocab, solution


