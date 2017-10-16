# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:09:42 2017

@author: Administrator
"""


from __future__ import print_function
import numpy as np
import pickle
from tree import tree
from scipy.stats import pearsonr
import datapre
import Embedding
import time


def getWordmap(textfile):#textfile is wordvector file
    words={}
    We = []
    f = open(textfile,'r',encoding='utf-8')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))            
            j += 1
        words[i[0]]=n#the words and index in words vector file
        We.append(v)#the word vector of every word,index
    return (words, np.array(We))

def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')#
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')#
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask

def lookupIDX(words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1

def getSeq(p1,words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    return X1

def getSeqs(p1,p2,words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    for i in p2:
        X2.append(lookupIDX(words,i))
    return X1, X2

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def getSimEntDataset(f,words,task):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 3:
                if task == "sim":
                    e = (tree(i[0], words), tree(i[1], words), float(i[2]))
                    examples.append(e)
                elif task == "ent":
                    e = (tree(i[0], words), tree(i[1], words), i[2])
                    examples.append(e)
                else:
                    raise ValueError('Params.traintype not set correctly.')

            else:
                print(i)
    return examples

def getSentimentDataset(f,words):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 2:
                e = (tree(i[0], words), i[1])
                examples.append(e)
            else:
                print(i)
    return examples



def sentiment2idx(sentiment_file, words):
    """
    Read sentiment data file, output array of word indices that can be fed into the algorithms.
    :param sentiment_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, golds. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location), golds[i] is the label (0 or 1) for sentence i.
    """
    f = open(sentiment_file,'r')
    lines = f.readlines()
    golds = []
    seq1 = []

    for i in lines:
        i = i.split("\t")
        p1 = i[0]; score = int(i[1]) # score are labels 0 and 1
        X1 = getSeq(p1,words)
        seq1.append(X1)
        golds.append(score)
    x1,m1 = prepare_data(seq1)
    return x1, m1, golds

def sim2idx(sim_file, words):
    """
    Read similarity data file, output array of word indices that can be fed into the algorithms.
    :param sim_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, x2, m2, golds. x1[i, :] is the word indices in the first sentence in pair i, m1[i,:] is the mask for the first sentence in pair i (0 means no word at the location), golds[i] is the score for pair i (float). x2 and m2 are similar to x1 and m2 but for the second sentence in the pair.
    """
    f = open(sim_file,'r')
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1,m1 = prepare_data(seq1)
    x2,m2 = prepare_data(seq2)
    return x1, m1, x2, m2, golds

def entailment2idx(sim_file, words):
    """
    Read similarity data file, output array of word indices that can be fed into the algorithms.
    :param sim_file: file name
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1, x2, m2, golds. x1[i, :] is the word indices in the first sentence in pair i, m1[i,:] is the mask for the first sentence in pair i (0 means no word at the location), golds[i] is the label for pair i (CONTRADICTION NEUTRAL ENTAILMENT). x2 and m2 are similar to x1 and m2 but for the second sentence in the pair.
    """
    f = open(sim_file,'r')
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = i[2]
        X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1,m1 = prepare_data(seq1)
    x2,m2 = prepare_data(seq2)
    return x1, m1, x2, m2, golds

def getWordWeight(weightfile, a=1e-3):
    if a <=0: # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weightfile) as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            if(len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])#compute the sum of word number
            else:
                print(i)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value/N)

def getWeight(words, word2weight):#
    weight4ind = {}
    for word, ind in words.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind

def seq2weight(seq, mask, weight4ind):
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i,j] > 0 and seq[i,j] >= 0:
                weight[i,j] = weight4ind[seq[i,j]]
    weight = np.asarray(weight, dtype='float32')
    return weight

def getIDFWeight(wordfile, save_file=''):
    def getDataFromFile(f, words):
        f = open(f,'r')
        lines = f.readlines()
        golds = []
        seq1 = []
        seq2 = []
        for i in lines:
            i = i.split("\t")
            p1 = i[0]; p2 = i[1]; score = float(i[2])
            X1, X2 = getSeqs(p1,p2,words)
            seq1.append(X1)
            seq2.append(X2)
            golds.append(score)
        x1,m1 = prepare_data(seq1)
        x2,m2 = prepare_data(seq2)
        return x1,m1,x2,m2

    prefix = "../data/"
    farr = ["MSRpar2012",
            "MSRvid2012",
            "OnWN2012",
            "SMTeuro2012",
            "SMTnews2012", # 4
            "FNWN2013",
            "OnWN2013",
            "SMT2013",
            "headline2013", # 8
            "OnWN2014",
            "deft-forum2014",
            "deft-news2014",
            "headline2014",
            "images2014",
            "tweet-news2014", # 14
            "answer-forum2015",
            "answer-student2015",
            "belief2015",
            "headline2015",
            "images2015",    # 19
            "sicktest",
            "twitter",
            "JHUppdb",
            "anno-dev",
            "anno-test"]
    #farr = ["MSRpar2012"]
    (words, We) = getWordmap(wordfile)
    df = np.zeros((len(words),))
    dlen = 0
    for f in farr:
        g1x,g1mask,g2x,g2mask = getDataFromFile(prefix+f, words)
        dlen += g1x.shape[0]
        dlen += g2x.shape[0]
        for i in range(g1x.shape[0]):
            for j in range(g1x.shape[1]):
                if g1mask[i,j] > 0:
                    df[g1x[i,j]] += 1
        for i in range(g2x.shape[0]):
            for j in range(g2x.shape[1]):
                if g2mask[i,j] > 0:
                    df[g2x[i,j]] += 1

    weight4ind = {}
    for i in range(len(df)):
        weight4ind[i] = np.log2((dlen+2.0)/(1.0+df[i]))
    if save_file:
        pickle.dump(weight4ind, open(save_file, 'w'))
    return weight4ind

def SeqandWords(We,seq,mask):#得到一个三维tensor，存储多个句子中每个词的词向量
    seq_words=np.zeros((seq.shape[0],seq.shape[1],We.shape[1]))#shape:10,31,50    
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i,j] > 0 and seq[i,j] >= 0:
                seq_words[i,j]=We[seq[i,j],:]
    seq_words = np.asarray(seq_words, dtype='float32')
    return seq_words   

def get_weighted_average(We, seq, weight):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i;->We.shape=(400001L,50L)
    :param x: x[i, :] are the indices of the words in sentence i  ->x.shape=(10L,31L)
    :param w: w[i, :] are the weights for the words in sentence i  ->w.shape=(10L,31L)
    :return: emb[i, :] are the weighted average vector for sentence i  ->emb.shape=(10L,50L)
    """
    n_samples = seq.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        #emb[i,:] = w[i,:].dot(We[x[i,:],:])
        emb[i,:] = weight[i,:].dot(We[seq[i,:],:]) / np.count_nonzero(weight[i,:])
    return emb

def sim_evaluate_all(We, words, weight4ind, scoring_function, params):
    prefix = "../data/"
    parr = []; sarr = []

    farr = ["MSRpar2012",
            "MSRvid2012",
            "OnWN2012",
            "SMTeuro2012",
            "SMTnews2012", # 4
            "FNWN2013",
            "OnWN2013",
            "SMT2013",
            "headline2013", # 8
            "OnWN2014",
            "deft-forum2014",
            "deft-news2014",
            "headline2014",
            "images2014",
            "tweet-news2014", # 14
            "answer-forum2015",
            "answer-student2015",
            "belief2015",
            "headline2015",
            "images2015",    # 19
            "sicktest",
            "twitter",
            "JHUppdb",
            "anno-dev",
            "anno-test"]

    for i in farr:
        p,s = sim_getCorrelation(prefix+i)
        parr.append(p); sarr.append(s)

    s = ""
    for i,j in zip(farr, parr):
        s += "%30s %10f\n" % (i, j)

    n = sum(parr[0:5]) / 5.0
    s += "%30s %10f \n" % ("2012-average ", n)

    n = sum(parr[5:9]) / 4.0
    s += "%30s %10f \n" % ("2013-average ", n)

    n = sum(parr[9:15]) / 6.0
    s += "%30s %10f \n" % ("2014-average ", n)

    n = sum(parr[15:20]) / 5.0
    s += "%30s %10f \n" % ("2015-average ", n)

    print (s)

    return parr, sarr

################################################
## for textual similarity tasks
################################################
def sim_getCorrelation(f,words,weight4ind):
    f = open(f,'r')
    lines = f.readlines()
    golds = []
    seq1 = []
    seq2 = []
    for i in lines:
        i = i.split("\t")
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = getSeqs(p1,p2,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1,m1 = prepare_data(seq1)
    x2,m2 = prepare_data(seq2)
    m1 = seq2weight(x1, m1, weight4ind)
    m2 = seq2weight(x2, m2, weight4ind)
    return x1,m1,x2,m2,golds
def getSentencesSeg():
    prefix = "data/"
    prefix1="data/dataseg/"

    farr = ["MSRpar2012",
            "MSRvid2012",
            "OnWN2012",
            "SMTeuro2012",
            "SMTnews2012", # 4
            "FNWN2013",
            "OnWN2013",
            "SMT2013",
            "headline2013", # 8
            "OnWN2014",
            "deft-forum2014",
            "deft-news2014",
            "headline2014",
            "images2014",
            "tweet-news2014", # 14
            "answer-forum2015",
            "answer-student2015",
            "belief2015",
            "headline2015",
            "images2015",    # 19
            "sicktest",
            "twitter"]

    for i in farr:
        f=open(prefix+i,'r')
        lines=f.readlines()
        fl1=open(prefix1+i+'-1.txt', 'w')
        fl2=open(prefix1+i+'-2.txt', 'w')
        fl3=open(prefix1+i+'-score.txt', 'w')
        for i in lines:
            i = i.split("\t")
            fl1.write(i[0])
            fl1.write("\n")
            fl2.write(i[1])
            fl2.write("\n")
            fl3.write(i[2])
        fl1.close()
        fl2.close()
        fl3.close()
    return 1
  
def sim_evaluate(emb1,emb2,golds):
    inn=(emb1*emb2).sum(axis=1)
    emb1norm=np.sqrt((emb1*emb1).sum(axis=1))
    emb2norm=np.sqrt((emb2*emb2).sum(axis=1))
    scores=inn/emb1norm/emb2norm
    
    for i,sc in enumerate(scores):
        if(sc=='nan' or np.isnan(sc)):
            print(sc,i)
    preds=np.squeeze(scores)
    p1=pearsonr(preds,golds)[0]  
    return p1
def saveResult(weight,filename):
    file=filename+'.txt'
    fl=open(file, 'w')
    for key, value in weight.items():
        key=str(key)
        value=str(value)
        s=key+"\t"+value
        fl.write(s)
        fl.write("\n")
    fl.close()
def AllCalculate(clause_weight,phrase_weight,word_weight,words,word_emb):
    prefix = "data/datapre/"

    farr1 = [
            "MSRpar2012-1.txt",
            #"MSRpar2012-2.txt",
            "MSRvid2012-1.txt",
            #"MSRvid2012-2.txt",
            "OnWN2012-1.txt",
            #"OnWN2012-2.txt",
            "OnWN2013-1.txt",
            #"OnWN2013-2.txt",
            "OnWN2014-1.txt",
            #"OnWN2014-2.txt",
            "SMTeuro2012-1.txt",
            #"SMTeuro2012-2.txt",
            "SMTnews2012-1.txt", # 4
            #"SMTnews2012-2.txt", 
            "FNWN2013-1.txt",
            #"FNWN2013-2.txt",
            "SMT2013-1.txt",
            #"SMT2013-2.txt",
            "headline2013-1.txt", # 8
            #"headline2013-2.txt",
            "headline2014-1.txt", # 8
            #"headline2014-2.txt",
            "headline2015-1.txt", # 8
            #"headline2015-2.txt",
            "deft-forum2014-1.txt",
           # "deft-forum2014-2.txt",
            "deft-news2014-1.txt",
           # "deft-news2014-2.txt",            
            "images2014-1.txt",
            #"images2014-02.txt",
            "images2015-1.txt",   # 19
           # "images2015-2.txt",
            "tweet-news2014-1.txt", # 14
           # "tweet-news2014-2.txt",
            "answer-forum2015-1.txt",
           # "answer-forum2015-2.txt",
            "answer-student2015-1.txt",
           # "answer-student2015-2.txt",
            "belief2015-1.txt",
            #"belief2015-2.txt", 
            "sicktest-1.txt",
           # "sicktest-2.txt",
            "twitter-1.txt"]
            #"twitter-2.txt"]
    farr2 = [
           # "MSRpar2012-1.txt",
            "MSRpar2012-2.txt",
           # "MSRvid2012-1.txt",
            "MSRvid2012-2.txt",
           # "OnWN2012-1.txt",
            "OnWN2012-2.txt",
           # "OnWN2013-1.txt",
            "OnWN2013-2.txt",
           # "OnWN2014-1.txt",
            "OnWN2014-2.txt",
            #"SMTeuro2012-1.txt",
            "SMTeuro2012-2.txt",
           # "SMTnews2012-1.txt", # 4
            "SMTnews2012-2.txt", 
           # "FNWN2013-1.txt",
            "FNWN2013-2.txt",
           # "SMT2013-1.txt",
            "SMT2013-2.txt",
           # "headline2013-1.txt", # 8
            "headline2013-2.txt",
            #"headline2014-1.txt", # 8
            "headline2014-2.txt",
           # "headline2015-1.txt", # 8
            "headline2015-2.txt",
           # "deft-forum2014-1.txt",
            "deft-forum2014-2.txt",
           # "deft-news2014-1.txt",
            "deft-news2014-2.txt",            
           # "images2014-1.txt",
            "images2014-02.txt",
           # "images2015-1.txt",   # 19
            "images2015-2.txt",
           # "tweet-news2014-1.txt", # 14
            "tweet-news2014-2.txt",
           # "answer-forum2015-1.txt",
            "answer-forum2015-2.txt",
           # "answer-student2015-1.txt",
            "answer-student2015-2.txt",
           # "belief2015-1.txt",
            "belief2015-2.txt", 
           # "sicktest-1.txt",
            "sicktest-2.txt",
           # "twitter-1.txt",
            "twitter-2.txt"]
            #"JHUppdb",
            #"anno-dev",
    farr_score = [
            "MSRpar2012-score.txt",
            #"MSRpar2012-2.txt",
            #"MSRvid2012",
            "MSRvid2012-score.txt",
            #"MSRvid2012-2.txt",
            "OnWN2012-score.txt",
            #"OnWN2012-2.txt",
            "OnWN2013-score.txt",
            #"OnWN2013-2.txt",
            "OnWN2014-score.txt",
            #"OnWN2014-2.txt",
            "SMTeuro2012-score.txt",
           #"SMTeuro2012-2.txt",
            "SMTnews2012-score.txt", # 4
            #"SMTnews2012-2.txt", 
            "FNWN2013-score.txt",
            #"FNWN2013-2.txt",
            "SMT2013-score.txt",
            #"SMT2013-2.txt",
            "headline2013-score.txt", # 8
            #"headline2013-2.txt",
            "headline2014-score.txt", # 8
            #"headline2014-2.txt",
            "headline2015-score.txt", # 8
            #"headline2015-2.txt",
            "deft-forum2014-score.txt",
            #"deft-forum2014-2.txt",
            "deft-news2014-score.txt",
            #"deft-news2014-2.txt",            
            "images2014-score.txt",
            #"images2014-02.txt",
            "images2015-score.txt",   # 19
            #"images2015-2.txt",
            "tweet-news2014-score.txt", # 14
            #"tweet-news2014-2.txt",
            "answer-forum2015-score.txt",
            #"answer-forum2015-2.txt",
            "answer-student2015-score.txt",
            #"answer-student2015-2.txt",
            #"answer-student2015",
            "belief2015-score.txt",
            #"belief2015-2.txt", 
            "sicktest-score.txt",
            #"sicktest-2.txt",
            "twitter-score.txt"]
            #"twitter-2.txt"]

    for file1,file2,score in zip(farr1,farr2,farr_score):
        sentence_file_1=prefix+file1
        sentence_file_2=prefix+file2
        
        golds=datapre.getGolds(score)

        sentence_list_1=datapre.DataPre(sentence_file_1)
        sentence_list_2=datapre.DataPre(sentence_file_2)
            
        t0=time.time()
        emb1=Embedding.Embedding(sentence_list_1,clause_weight,phrase_weight,word_weight,words,word_emb)
        emb2=Embedding.Embedding(sentence_list_2,clause_weight,phrase_weight,word_weight,words,word_emb)
        t1=time.time()
        print("embedding compute time: %f seconds" % round((t1 - t0),2))

        scores=sim_evaluate(emb1,emb2,golds)
        
