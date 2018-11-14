
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import WordPunctTokenizer
import matplotlib.pyplot as plt
import  nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from fuzzywuzzy import fuzz
import xgboost
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import gc


# In[5]:


gc.collect()


# In[6]:


pd.set_option('display.max_colwidth', -1)
trainFile = pd.read_csv(os.getcwd()+"\\train.csv")


# In[7]:


trainFile= trainFile[trainFile['question1'].map(str).apply(len)>8]


# In[8]:


trainFile =trainFile[trainFile['question2'].map(str).apply(len)>=10]


# In[9]:


trainFile.head(5)


# In[11]:



#text = nltk.Text([p_stemmer.stem(i).lower() for i in words if i not in stopwords.words()  ])

#lowercase_words = [word.lower() for word in words
#                  if word not in stopwords.words() and word.isalpha()]

tempDF= trainFile['question1'].apply(lambda x: str(x).lower())
trainFile['Q1Words']= tempDF.apply(WordPunctTokenizer().tokenize)

tempDF= trainFile['question2'].apply(lambda x: str(x).lower())
trainFile['Q2Words']= tempDF.apply(WordPunctTokenizer().tokenize)


# In[13]:


def intersection(a,b):
    exclude = set(b)
    sameWords = [x for x in set(a) if x in b]
    return sameWords
    
def union(a,b):
    AllWords= list(set(a)|set(b))
    return AllWords

def differ(a,b):
    exclude = set(b)
    different = [x for x in set(a) if x not in b]
    return different
    


# In[14]:





p_stemmer = PorterStemmer()
NotToRemove = ['again','against','before','same','between','further']
stops= differ(stopwords.words(),NotToRemove)
print(stops)
trainFile['Q1WordsStemmed'] = trainFile.apply(lambda x:[p_stemmer.stem(i) for i in x['Q1Words'] if i not in stops ],axis=1 )


# In[15]:


trainFile['Q2WordsStemmed'] = trainFile.apply(lambda x:[p_stemmer.stem(i) for i in x['Q2Words'] if i not in stops ],axis=1 )


# In[16]:


trainFile.head(2)


# In[28]:



trainFile['SameWords'] = trainFile.apply(lambda row:intersection(row['Q1Words'],row['Q2Words']) , axis=1)


# In[29]:


trainFile['DifferentWords'] = trainFile.apply(lambda row:differ(union(row['Q1Words'],row['Q2Words']),intersection(row['Q1Words'],row['Q2Words'])) , axis=1)
trainFile['DifferentWordsStopped'] = trainFile.apply(lambda row:differ(union(row['Q1WordsStemmed'],row['Q2WordsStemmed']),intersection(row['Q1WordsStemmed'],row['Q2WordsStemmed'])) , axis=1)


# In[30]:


trainFile['cosineSimiliarity'] = trainFile.apply(lambda row:len(row['SameWords'])/(len(union(row['Q1Words'],row['Q2Words']))) ,axis=1)


# In[31]:


trainFile[trainFile['DifferentWords'].map(str).apply(len) < 5 ]


# In[32]:


trainFile['POS'] = trainFile['DifferentWords'].apply(nltk.pos_tag)


# In[33]:


#trainFile['PartOfSpeech'] = trainFile['POS'].map('list').apply() if len(z)>1]
trainFile['PartOfSpeech'] = trainFile.apply(lambda x: [y[1] for y in x["POS"]],axis = 1)



# In[34]:


def count(lst):
    #print (i)
    #lst = b[a].loc[i]
    
    c=0
    #print(lst)
    if not lst:
        return 0
    #print(lst)
    for j in lst:
        if j in ['NN','NNS','NNP','NNPS']:
            c+=1
    return c        
#count (['VBZ','IN','JJ','NN','NN','NNPS','MS'])        
trainFile['NoOfNouns'] = trainFile.apply(lambda x: count(x['PartOfSpeech']) ,axis = 1)
#trainFile['NoOfNouns'] = trainFile.apply(count('PartOfSpeech' ))


# In[35]:


trainFile["tk_set_ratio"] = trainFile.apply(lambda x: fuzz.token_set_ratio(str(x["question1"]),str(x["question2"])),axis = 1)


# In[36]:


trainFile["ratio"] = trainFile.apply(lambda x: fuzz.ratio(str(x["question1"]),str(x["question2"])),axis = 1)


# In[37]:


trainFile["DiffLength"] = trainFile.apply(lambda x: abs(len(x["question1"])-len(x["question2"])), axis = 1)


# In[38]:


trainFile.head(2)


# In[220]:


trainFile.columns


# In[18]:


bins = np.linspace(0,1,50)
plt.hist(trainFile[trainFile["is_duplicate"] == 1]["cosineSimiliarity"],bins,alpha = 0.5,label = 'Duplicate')
plt.hist(trainFile[trainFile["is_duplicate"] == 0]["cosineSimiliarity"],bins,alpha = 0.5,label = 'Non-Duplicate')
plt.legend(loc='upper right')
plt.show()


# In[65]:


trainFile[(trainFile["is_duplicate"] == 1) & (trainFile["cosineSimiliarity"]<0.2) & (trainFile["NoOfNouns"]>10)]


# In[93]:


trainFile[(trainFile["is_duplicate"] == 0) & (trainFile["cosineSimiliarity"]>0.8) & (trainFile["NoOfNouns"]<2)]


# In[105]:


fuzz.ratio("When can I expect my Cognizant confirmation mail?","When can I expect Cognizant confirmation mail?")


# In[101]:


fuzz.ratio("What are the laws to change your status from a student visa to a green card in the US, how do they compare to the immigration laws in Canada?","What are the laws to change your status from a student visa to a green card in the US? How do they compare to the immigration laws in Japan?")


# In[19]:


grouped= trainFile.groupby(trainFile['SameWords'].str.len())['is_duplicate'].sum()


# In[257]:


grouped.describe()


# In[20]:


grouped.plot()


# In[28]:



grouped2 = trainFile.groupby(['NoOfNouns'])['is_duplicate']
#[grouped2.sum(),grouped2.count()]


# In[25]:


grouped2.sum().plot(kind="bar")


# In[64]:


bins = np.linspace(0,30,40)
plt.hist(trainFile[trainFile["is_duplicate"] == 1]["NoOfNouns"],bins,alpha = 0.5,label = 'Duplicate')
plt.hist(trainFile[trainFile["is_duplicate"] == 0]["NoOfNouns"],bins,alpha = 0.5,label = 'Non-Duplicate')
plt.legend(loc='upper right')
plt.show()


# In[290]:


X1=trainFile['cosineSimiliarity'].as_matrix()


# In[291]:


X2=trainFile['NoOfNouns'].as_matrix()
X2.shape


# In[111]:


X3=trainFile['tk_set_ratio'].as_matrix()


# In[112]:


X4=trainFile['ratio'].as_matrix()


# In[114]:


X4.shape


# In[144]:


X5 =trainFile["DiffLength"].as_matrix()


# In[292]:


X = np.column_stack((X1,X2,X3,X4,X5))


# In[146]:


X.shape


# In[47]:


Y=trainFile['is_duplicate'].as_matrix()
Y.shape


# In[57]:


MNB = MultinomialNB()
MNB.fit(X,Y)


# In[58]:


Y_pred = MNB.predict(X)
accuracy=(np.mean(Y_pred==Y))
print(accuracy)


# In[150]:


LR = LogisticRegression()
LR.fit(X,Y)
Y_LR = LR.predict(X)
accuracyLR=(np.mean(Y_LR==Y))
print(accuracyLR)


# In[151]:


log_loss(Y, Y_LR, eps=1e-15)


# In[67]:


#tokenizing with ngram
# comparing sentiment?
# LDA
# other people report

# Jaccard Coefficient


# In[152]:


LR.coef_


# In[153]:


LR.intercept_


# In[293]:



rf = RandomForestClassifier(n_estimators = 50, random_state = 0)
rf.fit(X,Y)
Y_rf = rf.predict(X)
accuracyRF = np.mean(Y==Y_rf)
accuracyRF


# In[294]:


log_loss(Y, Y_rf, eps=1e-15)


# In[163]:


log_loss(Y, Y_rf, eps=1e-15)


# In[118]:


log_loss(Y, Y_rf, eps=1e-15)


# In[79]:


log_loss(Y, Y_rf, eps=1e-15)


# In[109]:


confusion_matrix(Y, Y_rf)


# In[119]:


confusion_matrix(Y, Y_rf)


# In[124]:


confusion_matrix(Y, Y_rf)


# In[164]:


confusion_matrix(Y, Y_rf)


# In[ ]:


FP >FN


# In[295]:


GBD= GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=10)
GBD.fit(X,Y)
pred_GBD = GBD.predict(X)
np.mean(pred_GBD==Y)


# In[296]:


log_loss(Y, pred_GBD,eps=1e-15)


# In[160]:


confusion_matrix(Y, pred_GBD)


# In[170]:


xg = xgboost.XGBClassifier()
xg.fit(X,Y)
Y_xg = xg.predict(X)
np.mean(Y_xg==Y)


# In[297]:


num_topics = 1 #The number of topics that should be generated
passes = 10
story ="DeprecationWarning: The truth value of an empty array is ambiguous. Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty."
texts = [[word for word in story.lower().split()]]
#text = [word for word in story.lower().split()]
        
#corpus = corpus.replace("\\"," ")
#print(corpus)
dictionary   = corpora.Dictionary(texts)
#print(dictionary)

corpus = [dictionary.doc2bow(text) for text in texts]
lda = LdaModel(corpus,id2word=dictionary,num_topics=num_topics,passes=passes)
           


# In[305]:


num_topics = 1 #The number of topics that should be generated
passes = 10

def topic(lst):
    texts= [lst]
    dictionary   = corpora.Dictionary(texts)
    #print(dictionary)

    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus,id2word=dictionary,num_topics=num_topics,passes=passes)
   
    topics=[x[0] for x in lda.show_topic(0,topn=5)]
    return topics
    


# In[306]:


topic(['step', 'step', 'guid', 'invest', 'share', 'market', 'india', '?'])


# In[307]:


lda.show_topic(0,topn=5)


# In[39]:


differentWordsDiction ={}

def dictcount(lst, isdiff,differentWordsDiction):
    
      
    for word in lst:
        if word in differentWordsDiction:
            value0 = differentWordsDiction[word][0]
            value1 = differentWordsDiction[word][1]
            if isdiff:
                
                differentWordsDiction[word]=[value0,value1+1]
            else:     
                differentWordsDiction[word]=[value0+1,value1]
        else:
            if isdiff:
                
                differentWordsDiction[word]=[0,1]
            else:     
                differentWordsDiction[word]=[1,0]
            
            
    return  differentWordsDiction    

        
        
final=trainFile.apply(lambda x:dictcount( x['DifferentWordsStopped'], x['is_duplicate'],differentWordsDiction),axis=1)


# In[40]:


zeroOneList = final[0]


# In[41]:


len(zeroOneList)


# In[42]:


ls = []
for key,value in zeroOneList.items():
    ls.append((key,value[0],value[1]))


# In[43]:


ls_sorted = sorted(ls,key=lambda x: -x[1] - x[2])


# In[62]:


new_ls=[]

for x in ls:
    if (abs(x[1]-x[2])>300):
        if (x[1]/x[2] >7 or x[2]/x[1]>7):
            new_ls.append(x)


# In[63]:


len(new_ls)


# In[64]:


new_ls_sorted = sorted(new_ls,key=lambda x: abs(x[1] - x[2]), reverse=True)
#new_ls_sorted


# In[65]:


#[x for x in new_ls_sorted if x[0]=='again']

[x for x in ls if x[0]=='again']


# In[66]:


new_ls_sorted


# In[ ]:


#words making sense , will not lead to overfitting
again, enough,hour,small,full, accept, worth,like, use, differ, between, much, without, student, live, countri, name, long,show
relationship, join, hard,connect, %, $, size,, =


# In[67]:


def word_check(ls,word):
    if word in ls:
        return 1
    else:
        return 0


# In[70]:


trainFile['is_again'] = trainFile.apply(lambda x: word_check(x['DifferentWordsStopped'],'again'),axis = 1)


# In[72]:


trainFile['is_differ'] = trainFile.apply(lambda x: word_check(x['DifferentWordsStopped'],'differ'),axis = 1)


# In[73]:


trainFile['is_without'] = trainFile.apply(lambda x: word_check(x['DifferentWordsStopped'],'without'),axis = 1)


# In[74]:


trainFile['is_between'] = trainFile.apply(lambda x: word_check(x['DifferentWordsStopped'],'between'),axis = 1)


# In[75]:


trainFile['is_percentage'] = trainFile.apply(lambda x: word_check(x['DifferentWordsStopped'],'%'),axis = 1)


# In[76]:


trainFile['is_dollar'] = trainFile.apply(lambda x: word_check(x['DifferentWordsStopped'],'$'),axis = 1)


# In[77]:


trainFile['is_name'] = trainFile.apply(lambda x: word_check(x['DifferentWordsStopped'],'name'),axis = 1)


# In[78]:


trainFile.head(1)


# In[79]:


trainFile.columns


# In[80]:


X= trainFile.as_matrix(columns =['ratio','DiffLength','cosineSimiliarity','NoOfNouns', 'tk_set_ratio','is_again', 'is_differ', 'is_without',
       'is_between', 'is_percentage', 'is_dollar', 'is_name'])


# In[81]:


X.shape


# In[82]:


Y=trainFile['is_duplicate'].as_matrix()
Y.shape


# In[84]:



rf = RandomForestClassifier(n_estimators = 30, random_state = 0)
rf.fit(X,Y)
Y_rf = rf.predict(X)
accuracyRF = np.mean(Y==Y_rf)
accuracyRF


# In[233]:


diction = {'a':[1,2],'b':[3,2]}
diction['a']

diction['a'] = [1,3]
#if key in diction:
diction['a']  


# In[212]:


[1,3][1,2]


# In[264]:


trainFile[trainFile.apply(lambda x: 'again' in x['DifferentWords'],axis=1)]


# In[ ]:


a ="abc"
a.


# In[255]:


trainFile.columns


# In[329]:


import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)

