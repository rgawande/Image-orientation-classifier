# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:40:10 2019

@author: rgawande, rutture, ssarode
"""
def softmax(A):
    exps = np.exp(A - np.max(A, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

def relu(A):
    return np.maximum(0,A)

def relu_d(A):
    B = A
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > 0:
                B[i][j] = 1
            else: 
                B[i][j] = 0
    return B

def nnet_predict(data,y,output):
    
    with open(output, "r") as jsonfile:
        outer_list = json.load(jsonfile)
    w1,w2,w3,b1,b2,b3 = outer_list
    
    x = data
    l1 = np.dot(x,w1) + b1
    a1 = relu(l1)

    l2 = np.dot(a1,w2) + b2
    a2 = relu(l2)

    l3 = np.dot(a2,w3) + b3
    a3 = softmax(l3)
    yp = a3
    
    n = 0.00
    d = 0.00
    for yp, y in zip(yp,y):
        if yp.argmax() == y.argmax():
            n += 1
            d += 1
        else:
            d += 1
    accuracy = (n/d)*100
    return round(accuracy,2), a3

class NeuralNetwork(object):
    def __init__(self, inp, y, epochs, learning_rate, neurons, op):
        n = neurons
        self.w1  = np.random.rand(inp.shape[1],n)*np.sqrt(2./n)
        self.b1  = np.zeros((1, n))
        self.w2  = np.random.rand(n,n)*np.sqrt(2./n)
        self.b2  = np.zeros((1, n))
        self.w3  = np.random.rand(n,y.shape[1])*np.sqrt(2./y.shape[1])
        self.b3  = np.zeros((1, y.shape[1]))
        self.a   = learning_rate
        self.e   = epochs
        self.o = op
        pass
  
    
    def fit(self, inp, output):
        for i in range(self.e):

            x = inp
            y = output

            #forward propagation
            l1 = np.dot(x,self.w1) + self.b1
            a1 = relu(l1)

            l2 = np.dot(a1,self.w2) + self.b2
            a2 = relu(l2)

            l3 = np.dot(a2,self.w3) + self.b3
            a3 = softmax(l3)

            #backward propagation
            dl3 = a3-y
            dw3 = np.dot(a1.T,dl3)/x.shape[0]
            db3 = np.sum(dl3,axis=0,keepdims=True)/x.shape[0]

            dl2 = np.multiply(np.dot(dl3,self.w3.T),relu_d(l2))
            dw2 = np.dot(a1.T,dl2)/x.shape[0]
            db2 = np.sum(dl2,axis=0,keepdims=True)/x.shape[0]

            dl1 = np.multiply(np.dot(dl2,self.w2.T),relu_d(l1))
            dw1 = np.dot(x.T,dl1)/x.shape[0]
            db1 = np.sum(dl1,axis=0,keepdims=True)/x.shape[0]

            
            self.w3 -= self.a * dw3
            self.b3 -= self.a * db3
            self.w2 -= self.a * dw2
            self.b2 -= self.a * db2
            self.w1 -= self.a * dw1
            self.b1 -= self.a * db1
            
            outer_list = [self.w1.tolist(),self.w2.tolist(),self.w3.tolist(),self.b1.tolist(),self.b2.tolist(),self.b3.tolist()]
            with open(self.o, "w") as jsonfile:
                json.dump(outer_list, jsonfile)         
            # a, yp = nnet_predict(x,y,self.o)
            # print("\nEpoch -",i,": train accuracy: ",a,end=" ")
        outer_list = [self.w1.tolist(),self.w2.tolist(),self.w3.tolist(),self.b1.tolist(),self.b2.tolist(),self.b3.tolist()]
        with open(self.o, "w") as jsonfile:
            json.dump(outer_list, jsonfile)
        pass    

def decision_tree(n,d,o):
    def file_read(file_name):
      y_train1,X_train,names=[],[],[]
      with open(file_name,'r') as f:
        for l in f:
          names.append(l.split(" ")[0])
          y_train1.append(int(l.split(" ")[1]))
          X_train.append(list(int(i) for i in l.split(" ")[2:]))
      return names,X_train,y_train1
        
    def gen_data(X,y,training):
      import pandas as pd
      d=pd.DataFrame(X,columns=cols)
      d['orient']=y
      if training:
        d.to_csv("train.csv",index=False)
      else:
        d.to_csv("test.csv",index=False)
    
    def entropy(feature):
      import numpy as np
      unique=np.unique(feature)
      freq=[feature.count(i) for i in unique]
      s=sum(freq)
    
      l=[]
      for i in range(len(unique)):
          p=freq[i]/s
          l.append(-p*np.log2(p))
          
      return sum(l)
    
    def ID3(data,depth):
      import numpy as np  
      if (len(np.unique(data[label]))==1) or (len(data)<2) or (depth==max_depth):
        unique=np.unique(data[label])
        freq=[list(data[label]).count(i) for i in unique]
        return unique[np.argmax(freq)]
    
      else:
        depth+=1
    
        splits={}
        for attribute in attributes:
          import random     
          random.seed(0)
          splits[attribute]=[random.choice(list(data[attribute])) for i in range(nos)]
          
#        Referred from https://www.python-course.eu/Decision_Trees.php
#        Split referrence: https://github.com/SebastianMantey/Decision-Tree-from-Scratch/blob/master/notebooks/01.%20handling%20only%20continuous%20variables.ipynb        
        best_entropy=float('inf')
        for attribute in splits:
          for threshold in splits[attribute]:
            data_below=data[data[attribute]<=threshold]
            data_above=data[data[attribute]>threshold]
            n=len(data_below)+len(data_above)
            attribute_entropy=(len(data_below)/n)*entropy(list(data_below[label]))+(len(data_above)/n)*entropy(list(data_above[label]))
#         Referrence end    
            if attribute_entropy<=best_entropy:
              best_entropy=attribute_entropy
              best_attribute=attribute
              best_threshold=threshold
    
        data_below=data[data[best_attribute]<=best_threshold]
        data_above=data[data[best_attribute]>best_threshold]
    
        condition=best_attribute+"<="+str(threshold)
        tree={condition:[]}
    
        cond_true=ID3(data_below,depth)
        cond_false=ID3(data_above,depth)
    
        tree[condition].append(cond_true)
        tree[condition].append(cond_false)
    
        return tree
    
    def classify(data,tree):
      condition=list(tree.keys())[0]
      ind=condition.index('<')
      attribute,threshold=condition[:ind],condition[ind+2:]
    
      if data[attribute]<=int(threshold):
        sub_tree=tree[condition][0]
      else:
        sub_tree=tree[condition][1]
    
      if isinstance(sub_tree,dict):
        return classify(data,sub_tree)
      else:
        return sub_tree
    
    def main():
      import sys
      import numpy as np
      import pandas as pd
      global cols
      cols=[]
    #  print(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])
      if n=='train':
          _,X_train,y_train=file_read(d)
          
          for i in range(len(X_train[0])):
              cols.append('feat_'+str(i))
           # cols.append('orient')
          gen_data(X_train,y_train,True)    
        
          train=pd.read_csv("train.csv")
                
          global X,y,label,exp_info,max_depth,rows,attributes,nos
          X=train.iloc[:,:-1]
          y=train.iloc[:,-1]
          label='orient'
          rows=train.shape[0]
          max_depth=7
          attributes=X.columns
          nos=3
          
          with open(o,'w') as file:
            print(ID3(train,0), file=file)
      
      if n=='test':
          names,X_test,y_test=file_read(d)
          
          for i in range(len(X_test[0])):
              cols.append('feat_'+str(i))          
          gen_data(X_test,y_test,False)
          test=pd.read_csv("test.csv")
          y_test=test.iloc[:,-1]
        
          with open(o,'r') as f:
              dtree=eval(f.read())
        
          y_pred=[classify(test.iloc[i],dtree) for i in range(test.shape[0])]
          
          print("Testing Accuracy:",round(len([y_test[i] for i in range(len(y_test)) if y_test[i]==y_pred[i]])/len(y_test)*100,2))
          with open('output.txt','w') as file:
              for i in range(len(names)):
                  line=names[i]+" "+str(y_pred[i])
                  print(line, file=file)

    if __name__=="__main__":
        main()

def knn(n,d,o):
    def file_read(file_name):
      y_train,X_train,names = [],[],[]
      with open(file_name,'r') as f:
          for l in f:
              names.append((l.split(" ")[0]))
              y_train.append(int(l.split(" ")[1]))
              X_train.append(list(int(i) for i in l.split(" ")[2:]))
      return names,X_train,y_train
    
    import numpy as np
    if n=='train':
      with open(o,"w") as n:
        with open(d,"r") as f:
          n.write(f.read())
          
    elif n=='test':
      _,X_train,y_train=file_read(o)
      names,X_test,y_test=file_read(d)
      X_train = np.asarray(X_train)
      X_test = np.asarray(X_test)
      y_train = np.asarray(y_train)
      y_test = np.asarray(y_test)
    
      k = 10
      def eucd(x, y):
        return np.sqrt(np.sum((x-y)**2))
    
      pred = []
      for j in range(len(X_test)):
        d = {}
        for i in range(len(X_train)):
          dist = eucd(X_test[j], X_train[i]) 
          d[i] = (dist,y_train[i])
        dummy=[i[0] for i in d.values()]
        ind = np.argsort(dummy)[:k]
        n=[list(d.keys())[z] for z in ind]
        labels = [d[i][1] for i in n]
        pred.append(max(labels,key=labels.count))
      print('Testing accuracy:',(np.sum(pred==y_test)/len(y_test))*100)
      with open('output.txt','w') as file:
        for i in range(len(names)):
          line = names[i]+" "+str(pred[i])
          print(line, file=file)  
          
          
import numpy as np
import pandas as pd
import sys 
import json

n =sys.argv[1]
d =sys.argv[2]
output =sys.argv[3]
ch =sys.argv[4]


with open(d) as f:
    ncols = len(f.readline().split(' '))

x_data = np.loadtxt(d,usecols=range(2,ncols))
y_data = np.loadtxt(d,usecols=range(1,2))

if ch=='nnet':
    di = {0:0,90:1,180:2,270:3}
    for i in range(len(y_data)):
        y_data[i] = di[y_data[i]]    
    y_data = np.eye(4)[y_data.astype(int)]    
    x_data /= 255    
    x_data = (x_data - x_data.mean()) / x_data.std()
    
    if  n == 'test':
        acc, yp = nnet_predict(x_data,y_data,output)
        print("Testing Accuracy of Neural Network is: ",acc)

        names = []
        with open(d,'r') as f:
            for l in f:
                names.append((l.split(" ")[0]))
        
        t = []
        for i in yp:
            t.append(i.argmax())
            
        di = {0:0,1:90,2:180,3:270}
        
        with open('output.txt','w') as file:
            for i in range(len(names)):
                line = names[i]+" "+str(di[t[i]])
                print(line, file=file) 
    
    elif n == 'train':
        model = NeuralNetwork(x_data,y_data,50,0.1,128,output)
        model.fit(x_data,y_data)
        acc, yp = nnet_predict(x_data,y_data,output)
        print("\nTraining Accuracy of Neural Network is: ",acc)

elif ch == 'tree':
    decision_tree(n,d,output)
       
else:
    knn(n,d,output)
