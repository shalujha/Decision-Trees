#!/usr/bin/env python
# coding: utf-8

# #  Decision Trees
# -  Simple Tree like Structure, model makes decision at every node.
# -  useful in simple tasks
# -  one of the most popular algorithm
# -  Easy Explainability, easy to show how a decision process works

# # Why Decision Trees are Popular?
# -  easy to interpret and present.
# -  well defined logic, mimic human level thought.
# -  Random Forest, Ensembles of Decision Trees are more powrful classifiers.
# -  Feature values are preferred to be categorical, if the values are continous then they are discretized prior to building the model.

# # Build Decision Trees
# ### Two Common Algorithm
# -  CART(Classification and Regression Trees) --> uses gini Index (Classification) as metric.
# -  ID3(Iterative Dichotomiser 3)--> uses Entropy function and Information gain as metric

# # Entropy
# - This is the measure of randomness of system. 
# - Formula: 
# - Randomness is denoted by H(S)
# - H(S)= -sum over all classes (Pc log2 Pc)
# - where Pc is the probability of class c.
# - Lets understand this by an example : suppose we have a box containing  3 red and  3 blue balls and if we draw a red ball then Pc means what is the probability of drawing a red ball.
# - we can find out entropy by -((1/2 * log 2 * 1/2)+ (1/2* log2 * 1/2)) => +1 ans .
# - if you have only red ball or only green ball in the box then the probability will be 1 and in turn the entropy will be 0.
# - so entropy will be maximum when you have same no of examples in both the class.

# # Information Gain
# - It is basically difference in entropy.
# - IG(S,A)= H(S) - sum over all the new nodes [(Sv/S)* H(Sv)]
#  - where Sv/S is the ratio of number of examples belongs to particular node to total number of examples.

# - The Goal of the system is to maximise the information gain. that is we are minimising the new entropy.
# - we eventually want to reduce the entropy.
# - Information Gain helps us to decide which attribute should be used while co structing our decision tree.
# 

# # Decision Trees
# problem : Titanic Survivor Prediction Kaggle Prediction.
# # Learning Goals : 
# - How to preprocess data ?
# - Dropping not useful features.
# - Creating a Binary Search Tree from scratch.

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv('titanic.csv')
print(data.head())


# In[3]:


print(data.info())


# In[4]:


columns_to_drop=["PassengerId","Name","Ticket","Cabin","Embarked"]


# In[5]:


data_clean=data.drop(columns_to_drop,axis=1)
print(data_clean.head())


# Since our algorithm will going to accept only numerical data, then we have to convert our categorical data into numerical data

# In[6]:


from sklearn.preprocessing import LabelEncoder


# In[7]:


le=LabelEncoder()


# In[8]:


data_clean["Sex"]=le.fit_transform(data_clean["Sex"])


# In[9]:


print(data_clean.head())


# In[10]:


data_clean=data_clean.fillna(data_clean["Age"].mean())


# In[11]:


print(data_clean.info())


# In[12]:


# To access the data of particular row, use loc method:
data_clean.loc[2]


# In[13]:


input_cols=["Pclass","Sex","Age","SibSp","Parch","Fare"]
output_cols=["Survived"]
X=data_clean[input_cols]
Y=data_clean[output_cols]
print(X.shape,Y.shape)
print(type(X))


# In[14]:


def entropy(col):
    n=float(col.shape[0])
    counts=np.unique(col,return_counts=True)
    ent=0.0
    for ix in counts[1]:
        p=ix/n
        ent+=(-1.0*p*np.log2(p))
    return ent


# In[15]:


col=np.array([1,1,0,0,0,1])
print(entropy(col))


# In[16]:


def divide_data(x_data,fkey,fval):
    x_left=pd.DataFrame([],columns=x_data.columns)
    x_right=pd.DataFrame([],columns=x_data.columns)
    for ix in range(x_data.shape[0]):
        val=x_data[fkey].loc[ix]
        if val>fval:
            x_right=x_right.append(x_data.loc[ix])
        else:
            x_left=x_left.append(x_data.loc[ix])
    return x_left,x_right


# In[17]:


x_left,x_right=divide_data(data_clean[:10],"Sex",0.5)


# In[18]:


print(x_left)
print(x_right)


# In[19]:


def information_gain(x_data,fkey,fval):
    left,right=divide_data(x_data,fkey,fval)
    l=float(left.shape[0]/x_data.shape[0])
    r=float(right.shape[0]/x_data.shape[0])
    if left.shape[0]==0 or right.shape[0]==0:
        return -10000000000
    i_gain=entropy(x_data.Survived)-(l*entropy(left.Survived)+ r*entropy(right.Survived))
    return i_gain


# In[20]:


# Test our function
for ix in X.columns:
    print(ix)
    print(information_gain(data_clean,ix,data_clean[ix].mean()))


# if You have five examples in your dataset, and you make a very deep tree, then the decision tree is prone to over fitting, because instead of doing some kind of generalisation, it is doing predictions at each and every node. it basically means large death of decison tree is prone to over fitting and in turn poor generalisation. 
# parameters are the ones that the “model” uses to make predictions etc. For example, the weight coefficients in a linear regression model. Hyperparameters are the ones that help with the learning process. For example, number of clusters in K-Means, shrinkage factor in Ridge Regression
# 

# - if we plot a graph between no of nodes and accuracy, the training accuracy will increase always but test accuracy will increase to a  certain point after that it will start deviating.

# In[21]:


import matplotlib.pyplot as plt
img=plt.imread('graph.png')
plt.axis('off')
plt.imshow(img)
plt.show()


# from graph, we can see that the curve for training data goes on increasing and for test data it reach till certain point and after that it starts decreasing. so we will check till a fixed validation point where accuracy is maximum. so we will remove all those nodes which gives poor generalisation accuracy.This way is also called early stopping.

# In[22]:


class DecisionTree:
    # constructore
    def __init__(self,depth=0,max_depth=5):
        self.left=None
        self.right=None
        self.fkey=None
        self.fval=None
        self.depth=depth
        self.max_depth=max_depth
        self.target=None
    def train(self,X_train):
        features=["Pclass","Sex","Age","SibSp","Parch","Fare"]
        info_gains=[]
        for ix in features:
            ig=information_gain(X_train,ix,X_train[ix].mean())
            info_gains.append(ig)
        self.fkey=features[np.argmax(info_gains)]
        self.fval=X_train[self.fkey].mean()
        #print("Making Tree , Feature is "+ self.fkey)
        # split data
        left,right=divide_data(X_train,self.fkey,self.fval)
        left=left.reset_index(drop=True)
        right=right.reset_index(drop=True)
        # Truely a leaf node
        if left.shape[0]==0 or right.shape[0]==0:
            if Xtrain.survived.mean()>=0.5:
                self.target="Survived"
            else:
                self.target="Dead"
            return 
        # stop early when depth>= max_depth
        if self.depth>=self.max_depth:
            if X_train.Survived.mean()>=0.5:
                self.target="Survived"
            else:
                self.target="Dead"
            return
        # Recursive Case
        self.left=DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(left)
        self.right=DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(right)
        # You can set the target at every node:
        if X_train.Survived.mean()>=0.5:
            self.target="Survived"
        else:
            self.target="Dead"
        return 
    def predict(self,test):
        if test[self.fkey]>self.fval:
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)


# # Post Prunthing
# -   first grow the tree to the fullest, and starts reducing or removing those nodes that reduce accuracy or reduces generalisation.

# In[23]:


d=DecisionTree()
#d.train(data_clean)


# # Train-Validation-Test_Set-Split

# In[24]:


split=int(0.7*data_clean.shape[0])
train_data=data_clean[:split]
test_data=data_clean[split:]
test_data=test_data.reset_index(drop=True)
print(train_data.shape,test_data.shape)


# In[25]:


dt=DecisionTree()
dt.train(train_data)


# In[26]:


print(dt.fkey)
print(dt.fval)
print(dt.left.fkey)
print(dt.left.fval)
print(dt.right.fkey)
print(dt.right.fval)


# In[27]:


y_pred=[]
for i in range(test_data.shape[0]):
    y_pred.append(dt.predict(test_data.loc[i]))


# In[28]:


print(y_pred)


# In[29]:


y_actual=test_data[output_cols]


# In[30]:


print(y_actual)


# In[31]:


from sklearn.preprocessing import LabelEncoder


# In[32]:


le=LabelEncoder()
y_pred=le.fit_transform(y_pred)


# In[33]:


print(y_pred)


# In[34]:


print(y_pred.shape,y_actual.shape)


# In[35]:


y_pred=y_pred.reshape(-1,1)
print(y_pred.shape)


# In[36]:


acc=np.sum(np.array(y_pred)==np.array(y_actual))/y_pred.shape[0]


# In[37]:


print(acc)


# # Decision Tree using SKLearn

# In[38]:


from sklearn.tree import DecisionTreeClassifier


# In[39]:


dt=DecisionTreeClassifier(criterion="entropy",max_depth=5)


# In[40]:


dt.fit(train_data[input_cols],train_data[output_cols])


# In[41]:


dt.predict(test_data[input_cols])


# In[42]:


dt.score(test_data[input_cols],test_data[output_cols])


# # Visualising a Decision Tree

# In[45]:


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz


# In[47]:


dot_data=StringIO()
export_graphviz(dt,out_file=dot_data,filled=True,rounded=True)


# In[48]:


graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[ ]:





# In[ ]:




