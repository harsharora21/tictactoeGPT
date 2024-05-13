import numpy as np
import random
import os
import subprocess
from argparse import ArgumentParser

random.seed(1234)
nar=np.array
#prelim stats: around 40% legal moves after training for 3mins
#generalize this to arbitrary tic tac toe nxn
#convert it to randomly sample games

#generate random and ~~winning games separately~~
#to generate winning games just write a win detector and select it from there

#we assume board is n x n
def isWin(state,ch):
    n=state.shape[0]
    m=state==ch
    out = (m.sum(axis=0)==n).any() or (m.sum(axis=1)==n).any()
    out |= (m.diagonal().sum()==n) or (np.flipud(m).diagonal().sum()==n)
    return out

def genGame(state=None,ch=1,n=None):
    """
    x always moves first.
    empty is 0, x is 1, o is 2.
    This func will generate a random possible game staring from state
    with player ch. This function will modify the state variable.
    """
    game=[]
    if n!=None:
        state=np.zeros([n,n])
        game=[n]
    else:
        n=state.shape[0]
    
    if isWin(state,1) or isWin(state,2) or np.all(state!=0):
        return []
    
    poss = [(i,j) for i in range(n) for j in range(n) if state[i][j]==0]#possible games
    move_i, move_j = random.choice(poss) #randomly pick a move
    state[move_i][move_j]=ch
    game += [(move_i,move_j)]+genGame(state,2 if ch==1 else 1)
    return game

def convToAbc(game):
    return str(game[0])+''.join([chr(ord('a')+(x[0]*game[0] + x[1])) for x in game[1:]])+"$$$$$"

def convToA1(game):
    return str(game[0])+''.join([chr(ord('a')+x[0])+str(x[1]) for x in game[1:]])+"$$$$$"

def convToTxt(lst):
    return '\n'.join(map(str,lst))

def save(str):
    with open('./data/custom_char/input.txt','w') as f:
        f.write(str)
    return
def load(filename):
    with open(filename,'r') as f:
        out = f.read()
    return out

def board(game):
    n=game[0]
    x=np.zeros([n,n])
    for ch,(i,j) in enumerate(game[1:]):
        x[i][j]=1 if ch%2 else 2
    return x

def isLegal(game):
    n=game[0]
    x=np.zeros([n,n])
    for ch,(i,j) in enumerate(game[1:]):
        if i>=n or j>=n or x[i][j]!=0 or isWin(x,1) or isWin(x,2):
            return False
        x[i][j]=1 if ch%2 else 2
    return isWin(x,1) or isWin(x,2) or np.all(x!=0)

def processOutput(s):
    l=list(filter(lambda x:x!='',s.split('\n')))[4:]
    l2 = [l[i] for i in range(len(l)-1) if l[i][0]!='-' and l[i+1][0]!='-']
    return list(map(eval,l2))

def convFromAbc(s): #assumes single digit n
    n=int(s[0])
    game=[n]
    for x in s[1:].replace('$',''):
        c=ord(x)-ord('a')
        game.append(( int((c-(c%n))/n) , c%n))
    return game

def convFromA1(s): #assumes single digit n
    n=int(s[0])
    game=[n]
    ts = s[1:].replace('$','')
    for i in range(0,len(ts),2):
        c= (ord(ts[i])-ord('a'),int(ts[i+1]))
        game.append(c)
    return game

# command for training
# python3.11 train.py config/train_custom_char.py 
# more args --max_iters=5000 --lr_decay_iters=5000
parser = ArgumentParser()
parser.add_argument("-c","--createdata",action="store_true",help="Create Data")
args=parser.parse_args()

doanything = True
countLegalOnly = not args.createdata
useAbc = True
#convGame = convToA1
#convFromGame = convFromA1
if doanything:
    if countLegalOnly:
        #s=subprocess.check_output("cd /home/harsh/ThesisMSc/nanoGPT;python3.11 sample.py --out_dir=out-custom-char --seed=1337",shell=True)
        curdir = os.path.dirname(os.path.realpath(__file__))
        cmd = f"cd {curdir};python3.11 sample.py --out_dir=out-custom-char --seed=1337"
        cmd4 = f"cd {curdir};python3 sample.py --start=\\'4\\' --out_dir=out-custom-char --max_new_tokens=44 --num_samples=100"
        s=subprocess.check_output(cmd4,shell=True)
        s=s.decode('utf-8')
        if useAbc:
            l = list(map(convFromA1,filter(lambda x:x!='' and '$' in x,s.split('\n')[4:])) )
        else:
            l=processOutput(s)
        #l3 = list(filter(lambda x:x[0]==3,l))
        l4 = list(filter(lambda x:x[0]==4,l))
        print(f"percentage of legal moves: {sum(map(isLegal,l))/len(l)}")
        print(f"percentage of 4x4 legal moves: {sum(map(isLegal,l4))/len(l4)}")
    else:
        n=10**4
        per = 0.1 #percentage split between 3x3 and 4x4
        wper = [0,0,0,1,0.1]+[0.01]*5
        if useAbc:
            data=[]
            for k in range(10):
                data += [convToA1(genGame(n=k)) for i in range(int(wper[k]*n))]
        else:
            data = [genGame(n=3) for i in range(n)] + [genGame(n=4) for i in range(int(per*n))]
        random.shuffle(data)
        save(convToTxt(data))


# training using following
# python3 data/custom_char/prepare.py 
# python3.11 train.py config/train_custom_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0


# with abc 
# no legal moves in 4x4
# 20% legal moves in all
    

#without abc
# no legal moves at all (0%)

# with a1
# no legal moves in 4x4
# 1.4% legal moves in all
        
#improvement
# sampling
# encoding
# try 0,00,000-1,10,100,1000,-... encoding
# try fine tuning llama or gpt2
# ternary expansion
# q-learning
# negative examples
# string of states
# try a1, b1, b2 encoding as well


# simulate a bunch of games, halt after winning
# 2 encodings
    # a1, b1, a3 etc. (do train test split)
    # terneary vec enoding 
