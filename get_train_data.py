from datasets import load_dataset
import itertools
from collections import Counter

train_text = load_dataset("wikitext-103-v1", split="train")

all_text = [t['text'] for t in train_text]

word_freq(all_text)

with open('/data/full_train.txt', 'w') as f:
    for t in all_text:
        f.write(t)

def word_freq(corpus):

    corpus = [t.split() for t in corpus]
    corpus = list(itertools.chain.from_iterable(corpus))
    count_words = Counter(corpus)
    print('total count words', len(count_words))
    sorted_words = count_words.most_common()
    
    with open('word_freq.json', 'w') as f:
        json.dump(f, count_words)
