from __future__ import print_function
from model.learner import Learner
from model.solution import Solution
import numpy as np
# Contains the configuration for the network, and any module that is used in it
from tools.wordio import WordStream


class Model:
    def __init__(self,  input,              # flat file with text used for learning
                        inputrange=None,    # uses only the given range in the input file
                        inputstreamclass=WordStream, # reader for the input files
                        alpha=0.025,        # initial learning rate
                        build=[],           # functions called with (model) to construct the model
                        pipeline=[],        # python Pipe classes that process the input word generator
                        mintf=1,            # minimal collection frequency for terms to be used
                        windowsize=5,       # windowsize used for w2v models
                        iterations=1,       # number of times to iterate over the corpus when learning
                        threads=None,       # hard defines the number of threads used for learning, otherwise cores is used
                        cores=2,            # defines number of threads used for building
                        updatecacherate=0,  # after this number of processed words, the cached vectors and words are updated
                        updaterate=10000,   # after this number of processed words and alpha are updated
                        cachewords=0,       # number of most frequent words to cache to avoid memory collisions between threads
                        cacheinner=0,       # number of most frequent inner nodes to cache to avoid memory collisions between threads
                        downsample=0,       # parameter for downsampling frequent terms (0=no downsampling)
                        quiet=1,            # set to 1 to supress output
                        blockedmode=True,  # set to True to force the model finishing one iteration before continuing to the next
                        reg=0,
                        method='normal',    # 'normal' = no reg, 'reg' = add reg.
                        dataname='simlex',
                        **kwargs):
        self.__dict__.update(kwargs)
        self.input = input;
        self.inputrange = inputrange
        self.inputstreamclass = inputstreamclass
        self.alpha = alpha
        self.build = build
        self.pipeline = pipeline
        self.mintf = mintf
        self.windowsize = windowsize
        self.iterations = iterations
        self.updatecacherate = updatecacherate        # set to 10k when not caching
        self.updaterate = updaterate        # set to 10k when not caching
        self.cachewords = cachewords
        self.cacheinner = cacheinner
        self.downsample = downsample        # typical settings: 0, 10e-3 or 10e-5
        self.quiet = quiet
        self.blockedmode = blockedmode
        
        # Dennis 
        self.reg = reg
        self.method = method
        print('model method', self.method)
        print('downsampled dataset', dataname)
        print('epoch',iterations)
        self.dataname = dataname
        # -----

        # number of cores/threads to use in multithreading mode, by default for every core two
        # threads are used to overcome performance loss by memory blocks
        self.setThreads(threads if threads is not None else cores)
        self.cores = cores if cores is not None else self.threads

    # Builds the vocabulay, then build the learning pipeline, and push the inputs through the pipeline.
    def run(self):
        print('learner running...')
        Learner(self).run()

    def setThreads(self, threads):
        self.threads = threads
        self.tasks = threads

    # Used by vocabulary builders to store the result in the Model
    def setVocab(self, vocab):
        self.vocab = vocab
        # number of unique words in vocab, vocab.totalwords contains the total count
        self.vocsize = len(vocab)
        # print(self.vocsize)
        # output size of the w2v model, can be modified by other modules (e.g. HS trains against a Huffmann tree instead of the vocabulary)
        # Dennis
        target_words = []
        missing_words = []
        s0 = './select/'
        s1 = '-selected.txt'
        with open(s0 + self.dataname + s1) as f:
            lines = f.readlines( ) 
            for l in lines:
                l = l.strip()
                try:
                    word = self.vocab.get(l)
                    target_words.append(word.index)
                except:
                    missing_words.append(l)
        # print('missing word', missing_words)
        # exp1
        # print(self.vocab.keys())
        # orange = self.vocab.get('orange')
        # cat = self.vocab.get('cat')
        # print('cat', cat.index)
        # raise Exception('ok')
        # orange_id = orange.index
        # print(orange_id, 'orange')
        # exp1 
        # Dennis
        reg_weight = []
        min_freq = min(list(self.vocab.word_freq.values()))
        for i, value in enumerate(list(self.vocab.word_freq.values())):
            if value > 101:
                reg_weight.append(0)
            else:
                w = (value/50)+1.5
                # w = (value/30)+1.5
                w = 0.01/np.log(w)
                if w > 0.012:
                    reg_weight.append(0.012)
                else:
                    reg_weight.append(w)
        # print(reg_weight) 
        print('max weight',max(reg_weight))
        print('min weight', min(reg_weight))
        self.word_freq = np.array(reg_weight, dtype=np.float32)
        print('word freq size', len(self.word_freq))
        # ------
        
        self.outputsize = len(vocab)

    def setSolution(self, matrices):
        self.matrices = matrices                    # must store a Python reference to prevent garbage collection!
        self.getSolution().setSolution(matrices)    # since the solution is in Cython

    # return the solution, which is instantiated on first request, which should be after the vocabulary was build.
    def getSolution(self):
        if not hasattr(self, 'solution'):
            self.solution = Solution(self)
        return self.solution
