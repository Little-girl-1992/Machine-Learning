# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 15:17:58 2015
@author: thinkpad
"""
import matplotlib.pyplot as plt 
import tree

decisionNode = dict(boxstyle = "sawtooth",fc = "0.8")
leafNode = dict(boxstyle = "round4",fc = "0.8")
arrow_args = dict(arrowstyle="<-")

#叶节点的个数
def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':#判断节点的数据类型是否为字典
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1

	return numLeafs
#树的层数
def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else:
			thisDepth = 1

		if thisDepth > maxDepth : maxDepth = thisDepth
	return maxDepth
#预先存储数的信息
def retrieveTree(i):
	listOfTrees =[{'feather': {0: 'no', 1: {'fly': {0: {'egg': {0: 'no', 1: 'yes'}}, 1: 'yes'}}}}] 
	return listOfTrees[i]

def plotNode(nodeTex,centerPt,parentPt,nodeType):
	#标注函数():annotate,xy:标注的位置坐标,xytext:标注文本所在位置,
	#arrowprops：标注箭头属性信息,bbox:标注文本的框nodeType
	createPlot.ax1.annotate(nodeTex,xy = parentPt,xycoords = 'data',
                         xytext = centerPt,textcoords = 'data',
                         va = "center",ha="center",bbox=nodeType,
                         arrowprops=arrow_args,fontsize=16)

#在父子节点间填充文本信息
def plotMidTex(cntrPt,parentPt,txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTex):
	#计算宽与高
	numLeafs = getNumLeafs(myTree)
	#depth = getTreeDepth(myTree)
	firstStr = myTree.keys()[0]
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
	plotMidTex(cntrPt,parentPt,nodeTex)
	plotNode(firstStr,cntrPt,parentPt,decisionNode)
	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:
			plotTree.xOff=plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
			plotMidTex((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
	plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD

def createPlot(inTree):
	fig = plt.figure(1,facecolor = 'white')
	fig.clf()
	axprops = dict(xticks=[],yticks=[])
	#subplot返回一个次要情节轴定位通过给定的网格的定义
	createPlot.ax1 = plt.subplot(1,1,1,frameon=False,**axprops)
	plotTree.totalW=float(getNumLeafs(inTree))#存储树的宽度
	plotTree.totalD=float(getTreeDepth(inTree))#存储树的高度
	#已绘制节点的位置
	plotTree.xOff= -0.5/plotTree.totalW;
	plotTree.yOff= 1.0
	plotTree(inTree,(0.5,1.0),'')
	plt.show()	
