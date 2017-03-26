# -*-coding:utf-8-*-
from Tkinter import Tk, StringVar, Label, Entry, Button
import tkMessageBox
import operator
import matplotlib.pyplot as plt
from numpy import zeros, tile, array


def file2matrix(filename):
    fr = open(filename);  # 打开存放原始数据的文件
    arrayOfLines = fr.readlines();
    numberOfLines = len(arrayOfLines);  # 获取文件数据行数
    returnMat = zeros((numberOfLines, 3)) ;  # 创建一个numberOfLines行3列的纯0数组，用于保存格式化的数据
    returnLabel = [] ;  # 创建一个空列表，用于保存从数据文件中读取的每一行的分类标签
    index = 0;
    for line in arrayOfLines:
        line = line.strip();  # 去掉空格
        listFromLine = line.split('\t');  # 根据\t进行分隔
        returnMat[index, :] = listFromLine[0:3];  # 复制数据

        if (listFromLine[-1] == 'largeDoses'):
            tempLabel = 3;
        elif(listFromLine[-1] == 'smallDoses'):
            tempLabel = 2;
        elif(listFromLine[-1] == 'didntLike'):
            tempLabel = 1;
        returnLabel.append(tempLabel);    # 获取分类标签
        index += 1;
    return returnMat, returnLabel;


'''
作用：将待分类数据集与已有数据集以其标签进行计算，从而得出待分类数据集最有可能所属的类别
'''    
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 获取数据集的行数
    
    # 计算距离
    # tile(a,(b,c)):将a的内容在行上重复b次，列上重复c次
    # 下面这一行代码的结果是将待分类数据集扩展到与已有数据集同样的规模，然后再与已有数据集作差
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet 
    
    sqDiffMat = diffMat ** 2  # 对上述差值求平方   
    sqDistances = sqDiffMat.sum(axis=1)  # 对于每一行数据求和
    distances = sqDistances ** 0.5  # 对上述结果开方
    sortedDistIndicies = distances.argsort()  # 对开方结果建立索引 
    
    # 计算距离最小的k个点的Lable
    classCount = {}  # 建立空字典，类别字典，保存各类别的数目
    for i in range(k):  # 通过循环寻找k个近邻
        voteIlabel = labels[sortedDistIndicies[i]]  # 先找出开方结果索引表中第i个值对应的Label值
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 存入当前label以及对应的类别值
    
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 对类别字典进行逆排序，级别数目多的往前放
    
    # 返回结果
    return sortedClassCount[0][0]  # 返回级别字典中的第一个值，也就是最有可能的Label值

'''
作用：根据数据生成图

'''
def showPlot(datingDataMat,datingLabels):
    fig=plt.figure();
    ax=fig.add_subplot(1,1,1);
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels));
    plt.show();
def start():
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    showPlot(datingDataMat,datingLabels)
'''
作用：数据归一化
'''
def autoNormal(dataSet):
    minVal = dataSet.min(0);  # 求得每一列最小值
    maxVal = dataSet.max(0);  # 求得每一列最大值
    ranger = maxVal - minVal;  # 两者做差，得到差值矩阵
    # normalDataSet = zeros(shape(dataSet));  # 创建一个与dataSet规模相同的纯0矩阵
    m = dataSet.shape[0];  # 获取DateSet的数据行数
    normalDataSet = dataSet - tile(minVal, (m, 1));  # 将最小值矩阵扩展到m行，1列，再用DataSet去做差，结果存入normalDataSet中
    normalDataSet = normalDataSet / tile(ranger, (m, 1));  # 将最大值矩阵扩展到m行，1列，再用normalDataSet去除以差值矩阵，得到最终结果
    return normalDataSet, ranger, minVal;



'''
作用：从归一化数据集中抽出前100条做为测试数据，交给分类器分类，计算分类效果
'''
def test():
    hoRatio=0.1; #设置测试集在整体数据集中的比例
    mat,lab=file2matrix('datingTestSet.txt'); #文件格式转换
    normMat,ranges,minValues=autoNormal(mat); # 数据归一化
    m=normMat.shape[0]; # 获取数据行数
    numTestVecs=int(m*hoRatio); #获取测试数据行数，即1000*0.1=100
    errorCount=0.0; # 设置错误计数器
    for i in range(numTestVecs): # 遍历前100行数据，交给分类器进行分类，并计算错误率
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],lab[numTestVecs:m],3); # 将当前第i条数据交给分类器进行分类
        print('ML结果：',classifierResult,',实际结果：',lab[i]); # 显示当前行数据的分类结果
        if(classifierResult!=lab[i]): # 如果出错，错误计数器加1
            errorCount+=1.0;
    print('错误率：',(classifierResult/float(numTestVecs))*100,"%"); # 显示最终结果
test()
input()


'''
作用：用户输入一组信息，程序对其进行分类测试
'''
def classfifyPerson(flight, game, cream):
    resultList = ['屌丝', '文艺青年', '高富帅'];
    
    flight = float(flight);
    game = float(game);
    cream = float(cream);
    
    mat, lab = file2matrix('datingTestSet.txt');
    norMat, ranges, minVal = autoNormal(mat);
    arrays = array([flight, cream, game]);
    claResult = classify0((arrays - minVal) / ranges, norMat, lab, 3);
    
    # print('你的相亲极97%的可能性为：',resultList[claResult-1]);
    return '该人为：' + resultList[claResult - 1]



# 生成窗口
root = Tk()
root.title('相亲对象预测系统');

'''
作用：窗口居中方法，由于tkinter自身没有直接设置窗口居中的方法，只能来手工计算了.这也一个通用的窗口居中方法
参数：
  obj：为窗口对象
  w：为窗口宽度
  h：为窗口高度
'''
def center_window(obj, w, h):
    # get screen width and height
    ws = obj.winfo_screenwidth()
    hs = obj.winfo_screenheight()
    # calculate position x, y
    x = (ws / 2) - (w / 2)   
    y = (hs / 2) - (h / 2)
    obj.geometry('%dx%d+%d+%d' % (w, h, x, y))

center_window(root, 400, 300)  

# 设置窗口控件对应的变量类型及初始值
txtFlight = StringVar();
txtFlight.set('');
txtGame = StringVar();
txtGame.set('');
txtEat = StringVar();
txtEat.set('');
txtResult = StringVar();
txtResult.set('');


'''
作用：单击事件执行的方法，获取用户值，交给分类器，在Label中显示结果。
其中try块的作用检测用户输入值的合法性
'''
def submit(): 
    try:
        getFlight = float(txtFlight.get());
        getGame = float(txtGame.get());
        getEat = float(txtEat.get());
        
        resultStr = classfifyPerson(getFlight, getGame, getEat);    # 获取用户输入值并调用分类方法
       # tkMessageBox.showinfo('取值', txtFlight.get() + '--' + txtGame.get() + '--' + txtEat.get())
        txtResult.set(resultStr) ;    # 在Label中显示结果
        
    except(Exception):
        tkMessageBox.showwarning("警告", "每项值都不为空，且只能为实型！") ;
    

def closeWin():  # 退出按钮事件
    result = tkMessageBox.askokcancel(title='确认', message='确定要退出么亲？');    # 弹出确认窗口，捕获用户选择
    if(result == True):    # 如果用户选择是，则退出程序
        root.destroy();



'''
生成各控件
'''
labelFligt = Label(root, text=u'飞行里程：', font=(u"微软雅黑", 14), fg="blue");
labelFligt.grid(row=0, column=0, pady=10);
inputFlight = Entry(root, textvariable=txtFlight, font=(u"微软雅黑", 14), fg="blue");
inputFlight.grid(row=0, column=1, pady=10);

labelGame = Label(root, text=u'游戏时间：', font=(u"微软雅黑", 14), fg="blue");
labelGame.grid(row=1, column=0, pady=10);
inputGame = Entry(root, textvariable=txtGame, font=(u"微软雅黑", 14), fg="blue");
inputGame.grid(row=1, column=1, pady=10);

labelEat = Label(root, text=u'吃货程度：', font=(u"微软雅黑", 14), fg="blue");
labelEat.grid(row=2, column=0, pady=10);
inputEat = Entry(root, textvariable=txtEat, font=(u"微软雅黑", 14), fg="blue");
inputEat.grid(row=2, column=1, pady=10);

submitBtn = Button(root, text=u'判断一下', command=submit);    # 绑定事件
submitBtn.grid(row=3, column=0, pady=10);

labelResult = Label(root, text='', textvariable=txtResult, font=(u"微软雅黑", 14), fg="red");
labelResult.grid(row=4, column=0, columnspan=2, pady=10, sticky='e');

closeBtn = Button(root, text=u'关 闭', command=closeWin);
closeBtn.grid(row=5, column=2);

root.mainloop();
