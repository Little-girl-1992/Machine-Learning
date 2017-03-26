# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:19:22 2015

@author: thinkpad
"""
from numpy import *
#创建实验样本
def loadDataSet():
	postingList = [['my','dog','has','flea','problems','help','please'],
	['maybe','not','take','him','to','dog','park','stupid'],
	['my','dalmation','is','so','cute','I','love','him'],
	['stop','posting','stupid','worthless','garbage'],
	['mr','licks','ate','my','steak','how','to','stop','him'],
	['quit','buying','worthless','dog','food','stupid']]
	classVec = [0,1,0,1,0,1]
	return postingList,classVec

#创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)#求并集
	return list(vocabSet)
#统计一个词是否在词典中或重复多少词
def setOfWord2Vec(vocabList,inputSet):
	returnVec = [0] * len(vocabList)#创建一个和词汇表等长的向量且元素置0
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		else:	print("the word: is not in my Vocabulary!" % (word))
	return returnVec

def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)#文档数目
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = ones(numWords)#创建1矩阵
	p1Num = ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:	
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive
#判断类型归宿
def classifyNB(vec2Classify,p0Vect,p1Vect,pClass1):
	p1 = sum(vec2Classify * p1Vect) + log(pClass1)
	p0 = sum(vec2Classify * p0Vect) + log(1.0-pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNb(testEntry):
	listOposts,listClasses=loadDataSet()
	myVocabList = createVocabList(listOposts)
	trainMat=[]
	for PostinDoc in listOposts:
		trainMat.append(setOfWord2Vec(myVocabList,PostinDoc))
	p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
	thisDoc = array(setOfWord2Vec(myVocabList,testEntry))#要查找的词的向量
	print testEntry,'classified as :',classifyNB(thisDoc,p0V,p1V,pAb)
#解析文件
def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*',bigString)
	return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
	docList = []
	classList = []
	fullText = []
	#导入并解析文本文件
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt'%(i)).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)
		wordList = textParse(open('email/ham/%d.txt'%(i)).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)	
	vocabList = createVocabList(docList)#所有文件的不重复词
	trainingSet = range(50)
	testSet = []
	#随机构建10个文间的训练集
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	trainMat = []
	trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWord2Vec(vocabList,docList[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam) !=classList[docIndex]:
			errorCount += 1
	print 'the error rate is:',float(errorCount)/len(testSet)
spamTest()
