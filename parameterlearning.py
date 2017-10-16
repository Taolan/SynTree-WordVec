# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:21:51 2017

@author: Administrator
"""
 
import datapre,data_io,params,Embedding 
import random

################################################################
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
            "twitter-1.txt"
            #"twitter-2.txt"]
        ]
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
            "twitter-2.txt"

            ]
farr_score = [
            "MSRpar2012-score.txt",
            #"MSRpar2012-2.txt",
            #"MSRvid2012",
             "MSRvid2012-score.txt",
            #"MSRvid2012-2.txt",
           "OnWN2012-score.txt",
       #      #"OnWN2012-2.txt",
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
            "twitter-score.txt"
            #"twitter-2.txt"]
    ]
prints=""
rmpc = 1
params = params.params()
params.rmpc = rmpc
parr=[]
n_samples=0
for i in farr_score:
    golds=datapre.getGolds(prefix+i)
    n_samples+=len(golds)

samples = [[0 for col in range(3)] for row in range(n_samples)]
ind=0 
for file1,file2,scorefile in zip(farr1,farr2,farr_score):
    sentence_file_1=prefix+file1
    sentence_file_2=prefix+file2
    golds=datapre.getGolds(prefix+scorefile)

    sentence_list_1=datapre.DataPre(sentence_file_1)
    sentence_list_2=datapre.DataPre(sentence_file_2)
       
    for i,j,k in zip(sentence_list_1,sentence_list_2,golds):
        samples[ind][0]=i
        samples[ind][1]=j
        samples[ind][2]=k
        ind+=1
    
################################################################
#randomly select data 
samples=random.sample(samples,int(n_samples*0.6))
sentence1=[]
sentence2=[]
golds=[]
for i,item in enumerate(samples):
    sentence1.append(item[0])
    sentence2.append(item[1])
    golds.append(float(item[2]))

###################################################
Wordweight_file='weight/word_weight_3a.txt'
Clauseweight_file='weight/Clause_weight.txt'
Phraseweight_file='weight/Phrase_weight.txt'
word_weight=datapre.TreeNode_Weight(Wordweight_file)
clause_weight=datapre.TreeNode_Weight(Clauseweight_file)
phrase_weight=datapre.TreeNode_Weight(Phraseweight_file)

wordfile = 'wordvector/glove.6B.50d.txt' # word vector file, can be downloaded from GloVe website

(words, word_emb) = data_io.getWordmap(wordfile)

#####################################################

epoch=0.01
for key,we in word_weight.items(): 
    prescore=0.0
    score=0.0
    count1=0#add
    count2=0#sub
    result={}
    flag1=0
    flag2=0
    while count1<500:
        emb1=Embedding.Embedding(sentence1,clause_weight,phrase_weight,word_weight,words,word_emb,params)
        emb2=Embedding.Embedding(sentence2,clause_weight,phrase_weight,word_weight,words,word_emb,params)
        score=data_io.sim_evaluate(emb1,emb2,golds)        
        result[word_weight[key]]=score
        word_weight[key]+=epoch
        count1+=1
        if(score-prescore>0):
            prescore=score
        else:
            flag1+=1
            print("POS: %5s ; weight: %5s ;flag: %d ;count: %d"%(key,word_weight[key],flag1,count1))
            if(flag1>5):
                word_weight[key]=we
                break
    while count2<100:
        emb1=Embedding.Embedding(sentence1,clause_weight,phrase_weight,word_weight,words,word_emb,params)
        emb2=Embedding.Embedding(sentence2,clause_weight,phrase_weight,word_weight,words,word_emb,params)
        score=data_io.sim_evaluate(emb1,emb2,golds)
        result[word_weight[key]]=score
        word_weight[key]-=epoch
        count2+=1
        if(score-prescore>0):
            prescore=score
        else:
            flag2+=1
            print("POS: %5s ; weight: %5s ;flag: %d ;count: %d"%(key,word_weight[key],flag2,count2))
            if(flag2>5 or word_weight[key]<0):
                #find the biggest value of result
                word_weight[key]=sorted(result,key=lambda x:result[x])[-1]
                break
        
    #save the result
    #data_io.saveResult(result,'result/'+key)
print(word_weight)         
#data_io.saveResult(word_weight,'result/word_weight_4')        
