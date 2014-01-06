#-*- coding:utf-8 -*-   #允许文档中有中文
import sys
import numpy as np
import shutil
import sys
reload(sys)
sys.setdefaultencoding('utf-8') #允许打印unicode字符

DATASET = './data'

def isfloat(str):   #判断字符串是否是浮点数
    if str.isdigit():
        return True
    minpp = 1
    for i in range(0,len(str)):
        c = str[i]
        if c.isdigit():
            continue
        if c=='-' and i==0:
            minpp = 2
            continue
        if c=='.' and i>=minpp and i<(len(str)-1):
            continue
        return False
    return True


def extract_data_from_txt(txt,npy):  #从txt样本文件抽取数据转换为npy文件
    f = open(txt)
    x = []
    cnt = 0
    dim = 0 #维数

    for line in f:
        line = line.rstrip('\n')
        if len(line) < 1:
            continue
        if cnt == 0:  #第一行是每个点的维数
            dim = (int)(line)
            print 'Dimention:',line
            x = np.ndarray((0, dim), dtype='float64')
        else:   #余下的每行是每个点的值
            items = line.split()
            for i in range(0,len(items)):
                if isfloat(items[i]):
                    items[i] = float(items[i])
            if len(items)!=dim: #当前点的维数不正确
                print 'line:',cnt,'has',len(items),'items instead of',dim
                continue
            tmp = np.array([items])  
            tmp = tmp.astype('f')
            #print tmp
            x = np.append(x, tmp, axis=0)   #设置axis，否则会变成一维数组
        cnt += 1

    np.save(npy, x)
    
    print "length:", len(x)
    print "shape:", x.shape
    print ''

def extract_label_from_txt(txt,npy):  #从txt索引文件抽取标签转换为npy文件
    f = open(txt)
    y = []
    cnt = 0
    dim = 0 #维数
    pre = ''    #前一个数据点类名(初始使用数据点所在的文件名)
    now = ''    #当前数据点类名
    clus_cnt = 0    #类数

    for line in f:
        line = line.rstrip('\n')
        if len(line) < 1:
            continue
        if cnt == 0:  #第一行是每个点的维数
            dim = (int)(line)
            print 'Dimention:',line
        else:   #余下的每行是每个点的值
            items = line.split()
            #print items
            if len(items)!=dim: #当前点的维数不正确
                print 'line:',cnt,'has',len(items),'items instead of',dim
                continue
            now = items[1].decode('gbk')
            if pre != now:
                clus_cnt += 1
                pre = now
            #print now
            y.append(now)
            
        cnt += 1

    yy = np.array(y)
    np.save(npy, yy)

    print "length:", len(y)
    print "shape:", yy.shape
    print 'clus_cnt',clus_cnt
    print ''

if __name__ == '__main__':
    extract_data_from_txt('LSHVector.txt',DATASET+"/aligned_train_xdata.npy")
    extract_label_from_txt('LSHIndex.txt',DATASET+"/aligned_train_ylabels.npy")

    #复制文件
    # shutil.copyfile(DATASET+"/aligned_train_xdata.npy",DATASET+"/aligned_test_xdata.npy")
    # shutil.copyfile(DATASET+"/aligned_train_ylabels.npy",DATASET+"/aligned_test_ylabels.npy")
    #extract_from_txt('NLSHVector.txt','NLSHVector.npy')
    print 'Done'
