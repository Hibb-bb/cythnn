from arch.Word2Vec import Word2Vec
import argparse

from pipe.ConvertWordIds import convertWordIds
from pipe.DownSample import DownSample
from pipe.createInputTasks import createW2VInputTasks
from tools.word2vec import save
from tools.worddict import buildvocab
from pipe.ContextWindows import contextWindow

from model.model import Model


def get_args():
    parser = argparse.ArgumentParser(description='Cool Word2Vec')

    # data params
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)

    # training params
    parser.add_argument('--min-count', type=int, default=1)
    parser.add_argument('--window-size', type=int, default=5)
    parser.add_argument('--n-negs', type=int, default=5)
    parser.add_argument('--method', type=str, default='normal')
    parser.add_argument('--epoch', type=int, default=5)

    # emb dim
    parser.add_argument('--emb-dim', type=int, default=200)

    args = parser.parse_args()

    return args


args = get_args()
reg = 0
if args.method == 'reg':
    reg = 1
m = Model(alpha=0.025, vectorsize=args.emb_dim,
        input=args.data_path,
        inputrange=None,
        build=[buildvocab],
        pipeline=[ createW2VInputTasks, convertWordIds, DownSample, contextWindow, Word2Vec ],
        mintf=5, cores=1, threads=1, windowsize=args.window_size, downsample=0.001, iterations=args.epoch, negative=args.n_negs, reg=reg, method=args.method)
m.run()
save(args.save_dir, m, binary=False)
