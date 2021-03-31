from arch.SkipgramHScached import SkipgramHScached
from arch.SkipgramNScached import SkipgramNScached
from arch.SkipgramHS import SkipgramHS
from arch.SkipgramNS import SkipgramNS
from arch.CbowNS import CbowNS
from arch.CbowHS import CbowHS
from pipe.pipe import Pipe

# Behaves like a factory, returning the appropriate implementation in the transform method
class Word2Vec(Pipe):
    # default is Skipgram with Hierarchical Softmax (HS), unless configured differently
    # CBOW insteda iof Skipgram when model has cbow set to 1
    # use Negative Sampling (NS) instead of HS when negative > 0
    # Use HS caching when updatecacherate is set > 0
    def transform(self):
        # Dennis
        self.use_method = self.model.reg
        # ------

        if hasattr(self.model, 'cbow') and self.model.cbow == 1 and hasattr(self.model, 'negative'):
            return CbowNS(self.pipeid, self.learner)
        if hasattr(self.model, 'cbow') and self.model.cbow == 1:
            return CbowHS(self.pipeid, self.learner)
        if hasattr(self.model, 'negative'):
            if hasattr(self.model, 'cachewords') and self.model.cachewords > 0:
                # Dennis use_method
                return SkipgramNScached(self.pipeid, self.learner, self.use_method)
                # ------
            else:
                return SkipgramNS(self.pipeid, self.learner, self.use_method)
        if hasattr(self.model, 'cacheinner') and self.model.cacheinner > 0:
            return SkipgramHScached(self.pipeid, self.learner)
        return SkipgramHS(self.pipeid, self.learner)

