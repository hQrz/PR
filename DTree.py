import xlrd
import pickle
from math import log

def getDataSetAndLabel(path):#获得数据集和数据集标签
    xlsx=xlrd.open_workbook(path)
    dataSheet=xlsx.sheets()[0]
    dataSet=[]
    for row in range(1,dataSheet.nrows):
        rows=[]
        for col in range(8):
            rows.append(dataSheet.row_values(row)[col])
        dataSet.append(rows);
    label=[]
    for col in range(0,dataSheet.ncols):
        value=dataSheet.row_values(0)[col]        
        label.append(value)
    return dataSet,label
    
    
def calcEntropy(dataSet):#计算信息熵
    dataNum=len(dataSet)
    labelCount={}
    for vec in dataSet:
        if vec[-1] not in labelCount.keys():
            labelCount[vec[-1]]=0
        labelCount[vec[-1]]+=1
    entropy=0
    for key in labelCount:
        prob=float(labelCount[key])/dataNum
        entropy-=prob*log(prob,2)
    return entropy
  
def splitDataSet(dataSet,col,value):#获取数据集newDataSet={list\{list[col]} | list[col]=value,list in dataSet}
    newDataSet=[]
    for item in dataSet:
        if item[col]==value:
            info=[]
            info=item[:col]
            info.extend(item[col+1:])
            newDataSet.append(info)
    return newDataSet
    
def splitLabel(label,col):
    newLabel=label[:col]
    newLabel.extend(label[col+1:])
    return newLabel
    
  
def getDomain(dataSet,col):#获得某列的取值范围
    domain=[]
    for item in dataSet:
        if item[col] not in domain:
            domain.append(item[col])
    return domain
    
    
def selectBestLabel(dataSet,label):
    I=calcEntropy(dataSet)
    dataLen=len(dataSet)
    best=0
    maxGain=0
    bestDataSet=[]
    newLabel=[]
    for col in range(len(label)-1):
        E=0
        Gain=0
        tempDataSet=[]
        tempLabel=[]
        domain=getDomain(dataSet,col)
        for value in domain:
            newDataSet=splitDataSet(dataSet,col,value)
            tempLabel.append(value)
            tempDataSet.append(newDataSet)
            prob=len(newDataSet)/float(dataLen)
            E+=prob*calcEntropy(newDataSet)
        Gain=I-E
        if Gain>maxGain:
            maxGain=Gain
            best=col
            bestDataSet=tempDataSet
            newLabel=tempLabel
    return bestDataSet,label[best],newLabel
    
    
    
def dTree(dataSet,label):
    tree={}
    if len(dataSet[0])==1:#没有特征可遍历
        return dataSet[0][0]
    domain=getDomain(dataSet,-1)
    if len(domain)==1:#类别相同
        return domain[0]
    bestDataSet,bestLabel,bestLabels=selectBestLabel(dataSet,label)
    newLabel=splitLabel(label,label.index(bestLabel))
    nodes={}
    for Set in range(len(bestDataSet)):
        value=dTree(bestDataSet[Set],newLabel)
        nodes[bestLabels[Set]]=value
    tree[bestLabel]=nodes
    return tree
    
    
def storeTree(dTree,filename):
    fw=open(filename,'wb')
    pickle.dump(dTree,fw)
    fw.close()
    
    
def loadTree(filename):
    fr=open(filename,'rb')
    tree=pickle.load(fr)
    fr.close()
    return tree 
    
    
def vectorTest(vector,label,tree):
    keys=tree.keys()
    key=list(keys)[0]
    domain=tree[key]
    index=label.index(key)
    result='noting'
    for value in domain.keys():
        if value==vector[index]:
            if type(domain[value]).__name__=='dict':
                result=vectorTest(vector,label,domain[value])
            else:
                result=domain[value]
    return result
            
    
def dataSetTest(dataSet,label,tree):
    dataLen=len(dataSet)
    if dataLen==0:
        return 0
    rightNum=0
    for row in range(1,len(dataSet)):
        if vectorTest(dataSet[row],label,tree)==dataSet[row][-1]:
            rightNum+=1
    return rightNum/float(dataLen)
    
    
if __name__=='__main__':
    import time
    start=time.clock()
    #dataSet,label=getDataSetAndLabel('D:\\PR\\train.xlsx')
    #tree=dTree(dataSet,label)
    #storeTree(tree,'DTree.txt')
    tree = loadTree('DTree.txt')
    testDataSet,label=getDataSetAndLabel('D:\\PR\\test.xlsx')
    acc=dataSetTest(testDataSet,label,tree)
    print(acc,' %')
    print((time.clock()-start),' s')