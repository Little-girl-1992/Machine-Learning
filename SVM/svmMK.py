# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:26:43 2015

@author: thinkpad
"""
from numpy import *
#读文件得到类标签和整个数据矩阵
def loadDataSet(fileName):
	dataMat =[]
	labelMat=[]
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat,labelMat
#在0~m之间生成一个不等于i的数
def selectJrand(i,m):
	j=i
	while (j==i):
		j = int(random.uniform(0,m))
	return j
#返回一个H>=aj>=L的数
def clipAlpha(aj,H,L):
	if aj>H:
		aj=H
	if L>aj:
		aj=L
	return aj

#选择合适的第二个alpha值以保证在每次优化中采用最大步长
def selectJ(i,oS,Ei):
	maxK=-1
	maxDeltaE=0
	Ej=0
	oS.eCache[i]=[1,Ei]
	#构造一个非零列表，返回非零E值对应的alpha值
	validEcacheList = nonzero(oS.eCache[:,0].A)[0]
	if (len(validEcacheList))>1:
		for k in validEcacheList:
			if k == i:
				continue
			Ek=calcEk(oS,k)
			deltaE = abs(Ei -Ek)
			if (deltaE>maxDeltaE):
				maxK=k
				maxDeltaE=deltaE
				Ej=Ek
		return maxK,Ej
	else:
		j=selectJrand(i,oS.m)
		Ej=calcEk(oS,j)
	return j,Ej

def updataEk(oS,k):
	Ek=calcEk(oS,k)
	oS.eCache[k]=[1,Ek]

def kernelTrans(X,A,kTup):
	m,n=shape(X)
	K=mat(zeros((m,1)))
	if kTup[0]=='lin':
		K=X*A.T
	elif kTup[0]=='rbf':
		for j in range(m):
			#对行处理
			deltaRow = X[j,:] - A
			K[j] = deltaRow*deltaRow.T
		K = exp(K/(-1*kTup[1]**2))
	else:
		#raise引发一个你定义的异常
		raise NameError('Houston We Have a Problem --That Kernel is not recognized')
	return K
"""
修改后的程序
"""
class optStruct:
	"""docstring for optStructK"""
	def __init__(self, dataMatIn,classLabels,C,toler,kTup):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m,1)))
		self.b = 0
		self.eCache = mat(zeros((self.m,2)))
		self.K = mat(zeros((self.m,self.m)))
		for i in range(self.m):
			#对列处理
			self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)

#计算误差		
def calcEk(oS,k):
	fXk=float(multiply(oS.alphas,oS.labelMat).T*(oS.K[:,k]))+oS.b
	Ek=fXk - float(oS.labelMat[k])
	return Ek

def innerL(i,oS):
	Ei = calcEk(oS,i)
	#如果误差大，对该数据实例所对应的alpha值进行优化
	if ((oS.labelMat[i]*Ei< -oS.tol) and (oS.alphas[i]<oS.C)) or \
	((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
		j,Ej = selectJ(i,oS,Ei)
		#用于新老alphas比较
		alphaIold = oS.alphas[i].copy()
		alphaJold = oS.alphas[j].copy()
		if(oS.labelMat[i] != oS.labelMat[j]):
			L = max(0,oS.alphas[j] - oS.alphas[i])
			H = min(oS.C,oS.C+oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C,oS.alphas[j] + oS.alphas[i])
		if L==H: 
			print "L==H"
			return 0
		#最优修改量
		eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
		if eta>=0:
			print "eta>= 0"
			return 0
		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
		updataEk(oS,j)
		if (abs(oS.alphas[j] - alphaJold)<0.00001):
			print "j not moving enough"
			return 0
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		updataEk(oS,i)		
		b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - \
		oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
		b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - \
		oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
		if (0<oS.alphas[i]) and (oS.C>oS.alphas[i]):
			oS.b = b1
		elif (0<oS.alphas[j]) and (oS.C>oS.alphas[j]):
			oS.b=b2
		else:
			oS.b=(b1 + b2)/2.0
		return 1
	else:
		return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin', 0)):
	oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
	iter = 0
	entireSet = True;alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		if entireSet:
			for i in range(oS.m):
				alphaPairsChanged += innerL(i,oS)
			print "fullSet,iter: %d i:%d,pairs changed %d" %(iter,i,alphaPairsChanged)
			iter += 1
		else:
			nonBoundIs = nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print "non-bound,iter:%d i:%d,pairs changed %d" %(iter,i,alphaPairsChanged)
			iter += 1
		if entireSet:
			entireSet = False
		elif(alphaPairsChanged == 0):
			entireSet = True
		print "iteration number:%d"%(iter)
	return oS.b,oS.alphas
"""
手写识别问题改进
"""
def  img2vector(filename):
 	rows = 32
 	cols = 32
 	imgVector = zeros((1, rows * cols)) 
 	fileIn = open(filename)
 	for row in xrange(rows):
 		lineStr = fileIn.readline()
 		for col in xrange(cols):
 			imgVector[0, row * 32 + col] = int(lineStr[col])

 	return imgVector

def loadImages(dirName):
	from os import listdir
	hwLabels = []
	trainingFileList = listdir(dirName)
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		if classNumStr == 9:
			hwLabels.append(-1)
		else:hwLabels.append(1)
		trainingMat[i,:] = img2vector('%s/%s'%(dirName,fileNameStr))
	return trainingMat,hwLabels

def testDigits(kTup=('rbf',10)):
	dataArr,labelArr=loadImages('trainingDigits')
	b,alphas=smoP(dataArr,labelArr,200,0.0001,10000,kTup)
	datMat=mat(dataArr)
	labelMat=mat(labelArr).transpose()
	svInd=nonzero(alphas.A>0)[0]
	sVs=datMat[svInd]
	labelSV=labelMat[svInd]
	print "there are %d Support Vectors" %(shape(sVs)[0])
	m,n=shape(datMat)
	errorCount=0
	for i in range(m):
		kernelEval=kernelTrans(sVs,datMat[i,:],('rbf',kTup))
		predict=kernelEval.T*multiply(labelSV,alphas[svInd])+b
		if sign(predict)!=sign(labelArr[i]):
			errorCount += 1
	print "the training error rate is:%f" %(float(errorCount)/m)
	dataArr,labelArr=loadDataSet('testDigits')
	errorCount = 0
	datMat = mat(dataArr);labelMat=mat(labelArr).transpose()
	m,n=shape(datMat)
	for i in range(m):
		kernelEval=kernelTrans(sVs,datMat[i,:],('rbf',kTup))
		predict = kernelEval.T*multiply(labelSV,alphas[svInd])+b
		if sign(predict)!=sign(labelArr[i]):
			errorCount += 1
	print "the test error rate is: %f"%(float(errorCount)/m)

testDigits(('rbf',20))
