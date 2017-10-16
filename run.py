# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:21:51 2017

@author: Administrator
"""

import datapre
import data_io,params
import Embedding

Wordweight_file='weight/word_weight_3a.txt'
Clauseweight_file='weight/Clause_weight.txt'
Phraseweight_file='weight/Phrase_weight.txt'
word_weight=datapre.TreeNode_Weight(Wordweight_file)
clause_weight=datapre.TreeNode_Weight(Clauseweight_file)
phrase_weight=datapre.TreeNode_Weight(Phraseweight_file)

wordfile = 'wordvector/glove.6B.50d.txt' # word vector file, can be downloaded from GloVe website

(words, word_emb) = data_io.getWordmap(wordfile)
###########################################

prefix = "datapre/"

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
            #"images2014-2.txt",
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
            "images2014-2.txt",
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
prints=""
rmpc = 1
params = params.params()
params.rmpc = rmpc

parr=[]
for file1,file2,scorefile in zip(farr1,farr2,farr_score):
    sentence_file_1=prefix+file1
    sentence_file_2=prefix+file2
    golds=datapre.getGolds(prefix+scorefile)

    sentence_list_1=datapre.DataPre(sentence_file_1)
    sentence_list_2=datapre.DataPre(sentence_file_2)
            
    emb1=Embedding.Embedding(sentence_list_1,clause_weight,phrase_weight,word_weight,words,word_emb,params)
    emb2=Embedding.Embedding(sentence_list_2,clause_weight,phrase_weight,word_weight,words,word_emb,params)

    printstr=data_io.sim_evaluate(emb1,emb2,golds)
    
    parr.append(printstr)
#############################################
#print(parr)
sum2012=0
sum2013=0
sum2014=0
sum2015=0
sick=0
twitter=0
n12=0
n13=0
n14=0
n15=0
for i,j in zip(parr,farr1):
    prints+=j+" %10s\n" %(i)
    if(j.find('2012')!=-1):
        sum2012+=float(i)
        n12+=1
    elif(j.find('2013')!=-1):
        sum2013+=float(i)
        n13+=1
    elif(j.find('2014')!=-1):
        sum2014+=float(i)
        n14+=1
    elif(j.find('2015')!=-1):
        sum2015+=float(i)
        n15+=1
    elif(j.find('sicktest')!=-1):
        sick=float(i)
    elif(j.find('twitter')!=-1):
        twitter=float(i)
print(prints)   
print("STS2012:",sum2012/n12)
print("STS2013:",sum2013/n13)
print("STS2014:",sum2014/n14)
print("STS2015:",sum2015/n15)
print("Sicktest:",sick)
print("Twitter:",twitter)
