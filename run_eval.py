import torch
import argparse
from tqdm import tqdm
import numpy as np

from scipy.stats import pearsonr
from scipy.stats import spearmanr

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--GoldEmb', type = str, help = 'embedding location of gold embedding')
    parser.add_argument('--RegEmb', type = str, help = 'embedding location of our method (downsampled)')
    # parser.add_argument('--NorEmb', type = str, help = 'embedding location of normal skipgram (downsampled)')
    
    parser.add_argument('--OutLoc', type = str,default = './evaluation/result.txt' ,help = 'output location')

    return parser.parse_args()
    
class Eval:
    def __init__(self, args):

        self.args = args

        self.mae = True
    
        self.gold = self.read_emb(args.GoldEmb)

        self.reg = self.read_emb(args.RegEmb)

    def read_emb(self, dp):
        emb = {}
        with open(dp, 'r') as f:
            lines = f.readlines()
        f.close()
        for line in tqdm(lines[1:], total = len(lines)):
            word, freq, vec = line.split(' ', 2)
            wv = np.fromstring(vec, sep = ' ', dtype = 'float')
            wv = torch.from_numpy(wv)
            emb[word] = wv
        return emb
    
    def self_sim(self):
        with open('./word2vec/simlex-selected.txt','r') as f:
            lines = f.readlines()
        for i, l in enumerate(lines):
            lines[i] = l.strip()
        tgt = set(list(lines))
        cos = torch.nn.CosineSimilarity(dim = 0 , eps = 1e-6)

        tgt.remove('disorganize')
        tgt.remove('orthodontist')

        for t in tgt:
            if(("**"+t+"**") not in self.reg.keys()):
                # print('reg',t)
                pass
            if(("**"+ t+"**") not in self.gold.keys()):
                # print('gold',t)
                pass

        dif = [ cos( self.reg[w], self.reg['**'+w+'**']).item() for w in tgt]
        sg = [  cos( self.gold[w], self.gold['**'+w+'**']).item() for w in tgt]
        gold = [[1.00 for i in range(len(dif))]]
        
    def sre(self, v1, v2):
        pass
    
    def eval_simlex999(self):

        gold_res = []
        reg_res = []
        ans = []

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-5)
        print('simlex dataset')
        with open('./eval/SimLex-999.txt', 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            line = line.split()
            w1 = line[0]
            w2 = line[1]
            score = line[3]
            if w1 not in self.gold.keys() or w2 not in self.gold.keys():
                continue
            
            gold_sim = cos(self.gold[w1], self.gold[w2])
            reg_sim = cos(self.reg[w1], self.reg[w2])
            
            gold_sim = (gold_sim.item())
            reg_sim = (reg_sim.item())

            gold_res.append(gold_sim*100)
            reg_res.append(reg_sim*100)
            ans.append(float(score))


        gold_corr, _ = pearsonr(ans, gold_res)
        reg_corr, _ = pearsonr(ans, reg_res)

        sp_gold, _ =spearmanr(ans, gold_res)
        sp_reg, _ = spearmanr(ans, reg_res)

        print('*****************************************\nskip gram pearson correlation', gold_corr*100)
        print('*****************************************\nour method pearson correlation', reg_corr*100)

        print('*****************************************\nskip gram spearman correlation', sp_gold*100)
        print('*****************************************\nour method spearman correlation', sp_reg*100)

    def eval_men(self):
            
        scores = []
        
        cos = torch.nn.CosineSimilarity(dim = 0 , eps = 1e-6)
        print('MEN dataset')
        with open('./eval/pairs.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                w1, w2, s = line.split()
                s = float(s)/50
                scores.append([w1,w2,s])
        gold_sim = []
        reg_sim = []
        ans = []
        # self.gold, self.reg
        for w1,w2,s in scores:
            if(w1 not in self.gold.keys() or w2 not in self.gold.keys()):
                continue
            gold_sim.append(cos(self.gold[w1], self.gold[w2]).item())
            reg_sim.append(cos(self.reg[w1], self.reg[w2]).item())
            ans.append(s)

        gold_corr, _ = pearsonr(ans, gold_sim)
        reg_corr, _ = pearsonr(ans, reg_sim)

        sp_gold, _ =spearmanr(ans, gold_sim)
        sp_reg, _ = spearmanr(ans, reg_sim)

        print('*****************************************\nskip gram pearson correlation', gold_corr*100)
        print('*****************************************\nour method pearson correlation', reg_corr*100)

        print('*****************************************\nskip gram spearman correlation', sp_gold*100)
        print('*****************************************\nour method spearman correlation', sp_reg*100)
        

            


    def eval_rw(self):

        # scale = 4
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        
        ans = []
        gold = []
        reg = []
        normal = []
        pair_num =0

        with open('./data/rw/rw.txt', 'r') as f:
            lines = f.readlines()
            f.close()
        for line in tqdm(lines, total = len(lines)):
            line = line.split('\t')
            w1 = line[0]
            w2 = line[1]
            sim = line[2]
            if(w1 in self.gold.keys() and w2 in self.gold.keys()):
                pair_num += 1
                ans.append(torch.tensor(float(sim)))
                gold.append(cos(self.gold[w1], self.gold[w2])*100)
                reg.append(cos(self.reg[w1], self.reg[w2])*100)
                print(w1, w2, 'gold sim:',cos(self.gold[w1], self.gold[w2]))
                print(w1, w2, 'reg sim:',cos(self.reg[w1], self.reg[w2]))
                print(w1, w2, 'real sim:',sim)
                print('\n')




        ans = torch.stack(ans).numpy()
        gold = torch.stack(gold).numpy()
        reg = torch.stack(reg).numpy()

        gold_e = np.corrcoef(gold, ans)
        reg_e = np.corrcoef(reg, ans)

        print('\n=============d=========<Stanford Rare Words Dataset>=================\n')
        print('gold score :',gold_e)
        print('reg score :', reg_e)
        print('pair found in vocab:', pair_num)

        self.rw_gold_e = gold_e
        self.rw_reg_e = reg_e


    def eval_card(self):

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        
        # scale = 0-4

        self.card_gold = []
        self.card_reg = []
        self.card_nor = []
        score = []

        pair_num = 0

        with open('./data/card-660', 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            w1 = line[0]
            w2 = line[1]
            sim = line[2]
            if(w1 in self.gold.keys() and w2 in self.gold.keys()):
                pair_num+=1
                score.append( sim*2.5  )
                self.card_gold.append(cos(self.gold[w1], self.gold[w2])*100)
                self.card_reg.append(cos(self.reg[w1], self.reg[w2])*100)
                self.card_nor.append(cos(self.nor[w1], self.nor[w2])*100)

        self.card_gold = torch.stack(self.card_gold)
        self.card_reg = torch.stack(self.card_reg)
        self.card_nor = torch.stack(self.card_nor)
        score = torch.stack(score)


        if(self.mae == False):
            gold_e = ((self.card_gold-score)**2).mean()
            reg_e = ((self.card_reg-score)**2).mean()
            nor_e = ((self.card_nor-score)**2).mean()

        else:
            gold_e = (torch.abs(self.card_gold-score)).mean()
            reg_e = (torch.abs(self.card_reg-score)).mean()
            nor_e = (torch.abs(self.card_nor-score)).mean()

        print('\n============<Cambridge Rare Words Dataset>============\n')
        print('gold score :',gold_e)
        print('reg score :', reg_e)
        print('normal score :', nor_e)
        print('pair found in vocab:', pair_num)


args = get_args()
e = Eval(args)
            
e.eval_simlex999()
# e.eval_card()

# e.eval_men()       
# e.self_sim()
