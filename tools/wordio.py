import os, math, re, glob

# wraps a generator of words read from a flat text #file, within position #byterange
# uses virtual soft partitioning of flat text files, a partition starts after the first whitespace
# and the prior partition reads until the first word seperator after the boundary

# return filesize
def size(path):
    return os.path.getsize(path)

# reads the words that are in a flat text file, by a space or newline.
# The byterange indicates thedesignated range of words to be read.
# When the upper byterange boundary is set within a word or its first trailing word-separator
# the word is considered to be part of the range, when the lower byterange boundary is set
# within a word or its first trailing word-separator it is not.
# when window is set, an attempt is made to read #window words before and after
# the designated byterange, wentBack and wentPast indicate the number of words read before and
# after the designated boundary (max #window)
class WordStream:
    def __init__(self, file, windowsize, inputrange=None, eol=r'\n'):
        self.file = file
        self.windowsize = windowsize
        if input and inputrange is None:
            self.inputrange = range(0, size(input))
        else:
            self.inputrange = inputrange
        self.eol = eol

    def readFirst(self, f, bytepos, end):
        start = max(0, min(bytepos, bytepos - 1000))
        end = min(end, bytepos + 1000000)
        if start > 0:
            f.seek(start)
        buffer = f.read(end - start)
        pos = bytepos - start
        if self.windowsize > self.wentBack and bytepos > 0:
            while pos > 0 and self.windowsize > self.wentBack:
                pos -= 1
                if re.match(r'[\s' + self.eol + ']', buffer[pos]) or buffer[pos] == self.eol:
                    self.wentBack += 1
                    if self.windowsize == self.wentBack:
                        buffer = buffer[pos+1:]
            if pos == 0 and start > 0:
                self.wentBack += 1
        elif bytepos > 0:
            while pos < len(buffer):
                if re.match(r'\s', buffer[pos]) or buffer[pos] == self.eol:
                    break
                pos += 1
            buffer = buffer[pos+1:]
        return buffer

    def __iter__(self):
        buffer = ""
        self.wentPast = -1
        self.wentBack = 0
        with open(self.file, "r") as f:
            for chunk in chunkFixedSize(self.inputrange, 1000000):
                if chunk.start == self.inputrange.start and chunk.start > 0:
                    buffer = self.readFirst(f, chunk.start, chunk.stop)
                else:
                    newbuf = f.read(chunk.stop - chunk.start)
                    if not newbuf:
                        yield buffer
                        yield "</s>"
                        buffer = ""
                        break
                    buffer += newbuf
                for sentence in re.split('(' + self.eol + ')', buffer):
                    if re.match(self.eol, sentence):
                        yield buffer
                        yield '</s>'
                        buffer = ""
                    else:
                        words = re.split(r'\s', sentence)
                        for word in words[:-1]:
                            yield word
                        buffer = words[-1]
            newbuf = f.read((self.windowsize + 1) * 100)
            if not newbuf:
                if len(buffer) > 0:
                    yield buffer
                if self.inputrange.stop == size(self.file):
                    yield '</s>'
            else:
                buffer += newbuf
                for sentence in re.split('(' + self.eol + ')', buffer):
                    for word in re.split(r'\s', sentence):
                        self.wentPast += 1
                        yield word
                        if self.windowsize <= self.wentPast:
                            return
                    self.wentPast += 1
                    yield '</s>'
                    return

def loadStopwords(fname):
    result = set()
    with open(fname, 'r') as fin:
        for line in fin.readlines():
            line = line.strip()
            if len(line) > 0:
                result.add(line)
    return result


#setup a list of (file, range) tuples, for the given inputfiles
def inputUniform(inputfiles, threads, inputrange = None):
    listing = glob.glob(inputfiles)
    totalsize = 0
    for file in listing:
        totalsize += size(file)
    partsize = max(1000000, math.ceil(totalsize / threads))
    if partsize > 1000000:
        partsize = math.ceil((partsize + 1) / 2)

    chunks = []
    for file in listing:
        if inputrange is None:
            inputrange = range(0, size(file))
        subparts = math.ceil(len(inputrange) / partsize)
        for r in chunkRange(inputrange, subparts):
            chunks.append((file, r))
    return chunks

def inputDecay(inputfiles, threads, inputrange = None):
    listing = glob.glob(inputfiles)
    totalsize = 0
    for file in listing:
        totalsize += size(file)
    partsize = math.ceil(totalsize / threads)
    if partsize > 1000000:
        partsize = math.ceil((partsize + 1) / 2)
    chunks = []
    for file in listing:
        if inputrange is None:
            inputrange = range(0, size(file))
        subparts = math.ceil(len(inputrange) / partsize)
        for r in chunkRangeDecay(inputrange, subparts):
            chunks.append((file, r))
    return chunks

def chunkFixedSize(r, size):
    if not isinstance(r, range):
        r = range(r)
    return [ range(i, min(r.stop, i + size))
             for i in range(r.start, r.stop, size) ]

#split range in #n consecutive sub-ranges
def chunkRange(r, n):
    if not isinstance(r, range):
        r = range(r)
    step = math.ceil(len(r) / n)
    a = [ range(r.start + i * step, r.start + (i + 1) * step)
         for i in range(n - 1) ]
    a.append(range(r.start + (n - 1) * step, r.stop))
    return a

#split range in #n consecutive sub-ranges
def chunkRangeDecay(r, n):
    if not isinstance(r, range):
        r = range(r)
    step = math.ceil( 2 * (r.stop - r.start) / n / (n + 1))
    a = []
    start = r.start
    for incr in range(step, n * step, step):
        a.append(range(start, min(start + int(incr), r.stop)))
        start += int(incr)
    if start < r.stop:
        a.append(range(start, r.stop))

    return a
