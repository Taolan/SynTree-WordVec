# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:06:16 2017

@author: Administrator
"""

import datapre
import stack
import numpy as np
from sklearn.decomposition import TruncatedSVD
D_vector=50
stack=stack.Stack()

a= np.zeros((1, D_vector))
def Embedding(sentence_list,clause_weight,phrase_weight,word_weight,words,word_emb,params):
    emb= np.zeros((len(sentence_list), D_vector))
    for ind,sentence in enumerate(sentence_list):
        stack.set_size(len(sentence))
        sum_wordweight=getSumWeight(sentence,word_weight)
        for i,ch in enumerate(sentence):
            stack.push(ch)
            if(ch==")"):
                word_vec=np.zeros((1, D_vector)).astype("float32")
                we=0.0
                top=stack.top
                for index in range(-1,top):
                    word_pop=stack.pop()
                    if(word_pop==")"):
                        continue
                    elif(word_pop=="("):
                        if(type(word_vec)!=type(a) or type(we)!=type(1.0)):
                            print(ind,i,ch)
                            print(word_vec)
                            print(we)

                        if(we!=1.0):
                            we=we/sum_wordweight
                        if(we==0.0):
                            print(we,i,ch)
                        
                        word_vec=word_vec*we                    
                        stack.push(word_vec)
                        break
                    else:
                        if(type(word_pop)==type('a')):                                              
                            if(word_pop in word_weight):#the string is words or tags,if tags
                                we=float(word_weight[word_pop])
                            elif(word_pop in words):#if words
                                word_vec=datapre.Standardize(word_emb[words[word_pop]])                              
                                word_vec=np.array(word_vec,dtype=float)
                            elif((word_pop in phrase_weight) or (word_pop in clause_weight)):
                                we=1.0
                            elif(datapre.hasNumbers(word_pop)==True):#if number
                                word_vec=np.zeros((1,D_vector)).astype("float32")
                            else:
                                if(word_vec=="" or len(word_vec)<=0):
                                    word_vec=getAvgWordVectors(sentence,words,word_emb)
                                    print("the word %5s not exist in the file treenode_weight or glove.6B.50d,please add"%(word_pop))
                                elif(we==0.0):
                                    continue
                        elif(type(word_pop)==type(word_vec)):#if vector
                            if(word_vec.shape[1]==D_vector and word_vec!=''):
                                word_vec=word_pop+word_vec
                            else:
                                word_vec=word_pop
                        else:
                            print("ERROR:word_pop not string or np.nparray")
                            break
                        
                if(i==len(sentence)-1):
                    vec=stack.pop()
                    if(type(vec)==type(word_vec)):
                        emb[ind]=vec
                    else:
                        emb[ind]=word_vec
    if  params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb
def getSumWeight(sentence,weight):
    sumweight=0
    for ind,word in enumerate(sentence):
        if(word in weight):
            sumweight+=weight[word]
    return sumweight
def getAvgWordVectors(sentence,words,word_emb):
    sumwordvector=0
    num=0
    for ind,word in enumerate(sentence):
        if(word in words):
            sumwordvector+=word_emb[words[word]]
            num+=1
    return sumwordvector/num
            
def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX  
def compute_pc(X,npc=1):
    """
    Compute the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_      
               
        