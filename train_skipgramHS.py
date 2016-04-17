from convertWordIds.py import convertWordIds
from tools.taketime import taketime
from w2vContextWindows.cy import contextWindow
from w2vSkipgramHS.cy import trainSkipgramHS
from model.cy import model
from tools.nntools import createW2V, save
from tools.worddict import build_vocab
from tools.wordio import wordStreams
from w2vHSoftmax.cy import build_hs_tree

def doTestSkipgramHS(inputrange=None):
    return model(alpha=0.025, vectorsize=100,
                 input="data/text8",
                 inputrange=inputrange,
                 build=[ build_vocab, build_hs_tree, createW2V ],
                 pipeline=[ convertWordIds, contextWindow, trainSkipgramHS ],
                 mintf=5, cores=2, windowsize=5, iterations=1)

@taketime("run")
def time(m):
    m.run()

if __name__ == "__main__":
    m = doTestSkipgramHS()
    #m = doTestSkipgramHS( inputrange=range(10000) )

    time(m)

    save("results/vectors.sghsi5d0c2", m)
    save("results/vectors.sghsi5d0c2.bin", m, binary=True)

    print("done")

