# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 15:17:58 2015
@author: thinkpad
"""
import operator
from math import log
#计算给定数据集的香农熵
def calcShannonEnt (dataSet):
    numEntries = len(dataSet)#计算列表长度
    labelCounts = {}  # 建立空字典，类别字典，保存各类别的数目
    for featVec in dataSet:#取一行
        currentLabel = featVec[-1]#取一行的最后一个数据
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float (labelCounts[key])/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,1,'yes'],[1,1,0,'no'],[1,0,1,'yes'],[1,0,0,'no'],
               [0,1,1,'yes'],[0,1,0,'no'],[0,0,1,'no'],[0,0,0,'no']]
    labels = ['fly','egg','feather']
    return dataSet,labels

# 按照给定特征划分数据集，返回没有特征值的列表   
def splitDataSet(dataSet,axis,value):#待划分的数据集，划分的特征，特征的返回值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:#抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1#每一列数据的个数
    baseEntropy = calcShannonEnt(dataSet)#计算熵
    bestInfoGain = 0.0;bestFeature = 1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#迭代取每一列
        uniqueVals =set(featList)#建立一个集合
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {} # 建立空字典，保存各类别的出现的数目
    for vote in classList:
        #此标签如果不在字典里，放入并记录0
        if vote not in classCount.keys():classCount[vote] = 0 
        #此标签在字典里，记录+1
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
      key = operator.itemgetter(1),reverse = True)    
    # 返回值是一个对iterable中元素进行排序后的列表(list)
    # 1.返回访问元素的访问迭代器，
    # 2.key指定一个接收一个参数的函数，这个函数用于从每个元素中提取一个用于比较的关键字
    # 3.reverse是一个布尔值。如果设置为True，列表元素将被倒序排列。 
    return sortedClassCount[0][0]

#创建树的函数代码
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] #迭代取最后一列   
    # 统计在classList中出现的classList[0]的次数，
    # 第一个停止条件是所有类别相同
    if classList.count(classList[0]) == len (classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的
    #第二个停止条件是使用完了所有特征
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选择最好的数据集划分方式
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}#构建字典树
    del(labels[bestFeat])#删除指定项
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]#复制类标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:   classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)