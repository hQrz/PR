# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:42:43 2016

@author: Thinker
"""

import xlrd
from math import exp
from random import random
from numpy import array


def getDataSet(path):#获得数据集和数据集标签
    xlsx=xlrd.open_workbook(path)
    dataSheet=xlsx.sheets()[0]
    dataSet=[]
    for row in range(1,dataSheet.nrows):
        rows=[]
        for col in range(0,dataSheet.ncols):
            rows.append(dataSheet.row_values(row)[col])
        dataSet.append(rows);
    return dataSet
    
    
def sigmoid(x):
    if x>10:
        x=10
    else:
        if x<-10:
            x=-10
    e=exp(-x)
    num=1/float(1+e)
    return num


def rand():
    return (random()*2-1)


def suma(a):
    temp=0
    for i in a:
        temp+=i
    return temp
    
    
def creatANNs(inLayer,hiLayer,outLayer):
    anns=[]
    layer=[]
    theta=[]
    for bi in range(hiLayer):
        theta.append(rand())
        node=[]
        for xi in range(inLayer):
            node.append(rand())
        layer.append(node)
    anns.append(layer)
    layer=[]
    for yi in range(outLayer):
        node=[]
        for bi in range(hiLayer):
            node.append(rand())
        layer.append(node)
    anns.append(layer)
    gama=rand()
    return anns,theta,gama
    

def adjustANNs(vector,anns,theta,gama,step):
    annsOld=array(anns)
    yi=vector[-1]
    xk=array(vector[:-1])
    vk=annsOld[0]
    wk=annsOld[1][0]
    bk=[]
    yj=0
    temp=0
    #计算最后的结果yj
    for vi in range(len(vk)):
        temp=suma(xk*vk[vi])
        temp-=theta[vi]
        bk.append(sigmoid(temp))
    temp=suma(wk*array(bk))
    temp-=gama
    yj=sigmoid(temp)
    #调整ANNs
    deta=0
    for i in range(len(wk)):
        deta=(step*yj*(1-yj)*(yi-yj)*bk[i])
        wk[i]+=deta
    gama+=(step*yj*(1-yj)*(yi-yj)*-1)
    for vi in range(len(vk)):
        for vij in range(len(vk[vi])):
            deta=(step*bk[vi]*(1-bk[vi])*wk[vi]*yj*(1-yj)*(yi-yj)*xk[vij])
            vk[vi][vij]+=deta
        deta=(step*bk[vi]*(1-bk[vi])*wk[vi]*yj*(1-yj)*(yi-yj)*-1)
        theta[vi]+=deta
    for wi in range(len(anns[1][0])):
        anns[1][0][wi]=wk[wi]
    for vh in range(len(anns[0])):
        for vi in range(len(anns[0][vh])):
            anns[0][vh][vi]=vk[vh][vi]
    return anns,theta,gama,((yj-yi)*(yj-yi))/float(2)

  
def runANNs(dataSet):
    anns,theta,gama=creatANNs(len(dataSet[0])-1,5,1)
    while 1:
        E=0
        for vec in dataSet:
            anns,theta,gama,E=adjustANNs(vec,anns,theta,gama,0.01)
            if E<0.016:
                return anns,theta,gama
    return anns,theta,gama
    
   
def test(vector,anns,theta,gama):
    annsOld=array(anns)
    xk=array(vector[:-1])
    vk=annsOld[0]
    wk=annsOld[1][0]
    bk=[]
    #计算最后的结果yj
    for vi in range(len(vk)):
        temp=suma(xk*vk[vi])
        temp-=theta[vi]
        bk.append(sigmoid(temp))
    temp=suma(wk*array(bk))
    return sigmoid(temp-gama)
    
if __name__=='__main__':
    dataSet = getDataSet('D:\\PR\\bpTrainMin.xlsx')
    anns,theta,gama=runANNs(dataSet)
    testSet=getDataSet('D:\\PR\\bpTestMin.xlsx')
    testNum=len(testSet)
    rightNum=0
    for vec in testSet:
        y=test(vec,anns,theta,gama)
        if y-vec[-1]<0.35 and y-vec[-1]>-0.35:
            rightNum+=1
    print(float(rightNum/testNum))
