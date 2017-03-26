# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:03:27 2015

@author: thinkpad
"""
from numpy import *
#读出数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        #print lineArr
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
#公式
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
#求值
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) #转换为numpy矩阵数据类型            
    labelMat = mat(classLabels).transpose() #转置
    m,n = shape(dataMatrix)#行列
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              
        h = sigmoid(dataMatrix*weights) #矩阵相乘   
        error = (labelMat - h)              
        weights = weights + alpha * dataMatrix.transpose()* error 
    return weights
#画图
def plotBestFit(wei):
	import matplotlib.pyplot as plt
	weights = wei
	#dataMat,labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]#行数
	xcord1 = []
	ycord1 = []	
	xcord2 = []
	ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])			
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	x = arange(-4.0,4.0,0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]#最佳拟合直线
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()
#随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
	m,n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha * error * dataMatrix[i]
	return weights
#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter=150):
	m,n = shape(dataMatrix)
	weights = ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4/(1.0+j+i)+0.01
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			weights = weights + alpha * error * dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights
#以回归系数和特征向量作为输入来计算对应的Sigmoid值
def classifyVector(inX,weights):
	prob = sigmoid(sum(inX*weights))
	if prob>0.5:
		return 1.0
	else:
		return 0.0

def colicTest():
	frTrain = open('horseColicTraining.txt')#训练集
	frTest = open('horseColicTest.txt')#测试集
	trainingSet = []
	trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):#每行21个数据
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[21]))
	trainWeights = stocGradAscent1(array(trainingSet),trainingLabels)
	#前边是训练
	#后边是测试
	errorCount = 0
	numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
			errorCount += 1
	errorRate = (float(errorCount)/numTestVec)
	print ("the error rate of this test is: %f" %(errorRate))
	return errorRate

def multiTest():
	numTests = 10
	errorSum = 0.0
	for k in range(numTests):
		errorSum +=colicTest()
	print ("after %d iterations the average error rate is: %f" %(numTests,errorSum/float(numTests)))
multiTest()
	
