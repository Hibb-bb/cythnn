import argparse
import json
import pandas as pd
import random

parser = argparse.ArgumentParser('preprocessing')

parser.add_argument('--train_data', default='data/full_train.txt')
parser.add_argument('--tgt', default='word_sim/wordsim353.csv')
parser.add_argument('--ds_size',default=50)

args = parser.parse_args()

save_dir = 'data/'+ args.tgt.replace('word_sim/', '').replace('.csv', '') + f'drop{args.ds_size}.txt'

new_text=[]

drop_words = []

df = pd.read_csv(args.tgt)
for i, row in df.iterrows():
    drop_words.append(row[0])
    drop_words.append(row[1])

drop_words = list(set(drop_words))

with open('word_freq.json') as f:
    word_freq = json.load(f)

new_file=[]
with open(args.train_data) as f:
    for line in f.readlines():
        tokens = line.split()
        sent = ''
        for t in tokens:
            if t in word_freq.keys() and random.random() < args.ds_size/word_freq[t]:
                continue
            else:
                sent = sent + ' ' + t
        new_file.append(sent)

with open(save_dir, 'w') as f:
    for line in new_file:
        f.write(lines.replace('\n', ' '))
