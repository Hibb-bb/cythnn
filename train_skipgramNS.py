from arch.Word2Vec import Word2Vec
from pipe.ConvertWordIds import convertWordIds
from model.model import Model
from pipe.DownSample import DownSample
from pipe.createInputTasks import createW2VInputTasks
from tools.word2vec import save
from tools.worddict import buildvocab
from pipe.ContextWindows import contextWindow
import argparse


parser = argparse.ArgumentParser()
   
parser.add_argument('--train_data', type=str, default='data/test.txt',
    help="training data file")
parser.add_argument('--windowsize', type=int, default=5)
parser.add_argument('--method', type=str, default='reg', help="normal|reg")
parser.add_argument('--neg', type=int, default=5)
parser.add_argument('--emb', type=int, default=100)
args = parser.parse_args()


# Word2Vec uses Skipgram by default, set negative > 0 to use negative sampling instead of Hierarchical Softmax
if __name__ == "__main__":
    m = Model(alpha=0.025, vectorsize=args.emb,
                 input=args.train_data,
                 inputrange=None, # means all
                 build=[ buildvocab ],
                 pipeline=[ createW2VInputTasks, convertWordIds, DownSample, contextWindow, Word2Vec ],
                 mintf=5, cores=1, threads=1, windowsize=5, downsample=0.001, iterations=5, negative=args.neg, reg=1, method=args.method)
    m.run()
    save("results/vectors.sgns.bin", m, binary=True)
# 
