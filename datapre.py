# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:59:44 2017

@author: Administrator
"""
import re
import collections
import numpy as np

remove='[’!"#$%&\'*+,-./:;<=>?@[\\]^_`{|}~]+'
def StrSplit(data,strsplit):
    result=""
    string=data.split(strsplit)
    if(string==""):
        return result
    for s in string:
        if(s!=""):
            result=s
    return result
def DataPre(file):#change the string of sentences to array  
    datafile=open(file, 'r',encoding='utf-8')
    lines=datafile.readlines()
    rs = []
    for i in lines:
        i = i.split()

        temp=[]
        for j in i:
            flag=1
            s=re.sub(remove,'',j)
            if((s==")" or s=="(")and j!=s):#brackets with special string
                continue
            elif(j!=s and (j.count(")")>1 or j.count("(")>1) and hasLetter(s)!=True and hasNumbers(s)!=True):#包含了其他标点的多个括号的字符串
                    count=j.count(")")-1
                    j=j[-count:]
            for index,ch in enumerate(j):                
                if(ch=="(" and index==0):#only one left bracket
                    temp.append(ch)
                    temp.append(StrSplit(j,"("))
                elif(ch==")"):                
                    if(j.count(")")==1):#only one right bracket
                        temp.append(StrSplit(j,")"))
                    elif(j.count(")")>2 and flag==1):#more than two right brackets
                        temp.append(StrSplit(j,")"))
                        flag=0
                    elif(j.count(")")==2):#two right brackets
                        s_append=StrSplit(j,")")
                        if(index!=len(j)-1 and s_append!=""):
                            temp.append(s_append)
                    temp.append(ch)
                else:
                    continue
        rs.append(temp)
    return rs
def TreeNode_Weight(file):
    data=open(file,'r')
    lines=data.readlines()
    weight={}
    for (n,i) in enumerate(lines):
        i=i.split()
        weight[i[0]]=float(i[1])#save the words and weights(weight['IP']=1)
    return weight
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
def hasLetter(inputstring):
    r = re.compile(r'^[a-zA-Z]')
    rs=False
    for item in inputstring:
        result = r.match(item)
        if result != None:
            rs=True#have letter
            break
        else:
            continue
    return rs
def getSentencesSeg(f):
    f=open(f,'r')
    lines=f.readlines()
    fl1=open('MSRpar2012-1.txt', 'w')
    fl2=open('MSRpar2012-2.txt', 'w')
    fl3=open('MSRpar2012-score.txt', 'w')
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
def removePunctuation(file):#remove the punctuation in sentences except '.' at the end of sentence
    r='[!"#$%&\'()*+=,-./:;<=>?@[\\]^_`{|}~]+'
    f=open(file,'r')
    lines=f.readlines()
    f_save=open('data/data-2.txt','w')
    for i in lines:
        i=i.replace("'s","TAOLANUKUK")#repalce the ''s' with special string to retain the 's
        i=re.sub(r,'',i)
        i+='.'
        i=i.replace("TAOLANUKUK","'s")
        f_save.write(i)
        f_save.write("\n")
    f_save.close()
def removeSpecialPunctuation(file):
    #remove the special string(. .)
    f=open(file,'r')
    lines=f.readlines()
    f_save=open('data/data-1-parse-datapre.txt','w')
    for i in lines:
        i=i.replace(" (. .)","")
        f_save.write(i)
    f_save.close()
          
def getGolds(file):
    f = open(file,'r')
    lines = f.readlines()
    golds = []
    for i in lines:
        i=i.strip()
        golds.append(float(i))
    return golds
def SentenceCount(sentencefile,weight):
    r='[!"#$%&\'()*+=,-./:;<=>?@[\\]^_`{|}~]+'
    with open(sentencefile) as file1:
        str1=file1.read()
        str1=re.sub(r,'',str1)
        str1=str1.split()

    counter = collections.Counter(str1)#count the number of every word
    weight_word={}
    sumall=0
    for word in counter:
         if(word.isupper()):
             weight_word[word]=counter[word]
             sumall+=counter[word]
    sumall=sumall-weight_word['ROOT']-weight_word['S']-weight_word['SBARQ']-weight_word['SINV']
    for word in weight_word:
        if(word=='ROOT' or word=='S' or word=='SBARQ' or word=='SINV'):
            weight_word[word]=1
        else:
            weight_word[word]=weight_word[word]/sumall
    return weight_word
def NormalizedWordWeight(word_weight):
    sum_word_weight=0
    nor_word_weight={}
    for key,value in word_weight.items():
        sum_word_weight+=value
    for key,value in word_weight.items():
        nor_word_weight[key]=value/sum_word_weight
    return nor_word_weight
def Standardize(seq):
  #subtract mean
    centerized=seq-np.mean(seq, axis = 0)
  #divide standard deviation
    normalized=centerized/np.std(centerized, axis = 0)
    return normalized 