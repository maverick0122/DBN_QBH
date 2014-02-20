#-*- coding:utf-8 -*-   #允许文档中有中文
"""
将原始txt数据文件转化为npy文件，作为dbn输入文件

包括：
1.转换训练数据的txt文件到npy文件
2.转换标签索引的txt文件到npy文件
"""

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
            tmp = tmp.astype('f')   #数据转换为float类型
            #print tmp
            x = np.append(x, tmp, axis=0)   #设置axis，否则会变成一维数组
        cnt += 1

    np.save(npy, x)
    
    print "length:", len(x)
    print "shape:", x.shape
    print ''

def extract_filename(fname):    #取文件名，去除文件路径和后缀
    left = fname.rfind('\\')+1
    right = fname.rfind('.')
    if left <= 0:
        left = 0
    if right < 0:
        right = len(fname)

    return fname[left:right]


def extract_label_from_txt(txt,npy,clus_col):  #从txt索引文件抽取标签转换为npy文件,clus_col为聚类属性所在的列,一般为文件名所在的列
    f = open(txt)
    y = []
    cnt = 0
    dim = 0 #维数
    clus = {}
    now = ''    #当前数据点类名

    for line in f:
        line = line.rstrip('\n')
        if len(line) < 1:
            continue
        if cnt == 0:  #第一行是每个点的维数
            dim = (int)(line)
            print 'Dimention:',line
            if clus_col >= dim:
                print 'cluster column exceed Dimention.'
                sys.exit(-1)

        else:   #余下的每行是每个点的值
            items = line.split()
            #print items
            if len(items)!=dim: #当前点的维数不正确
                print 'line:',cnt,'has',len(items),'items instead of',dim
                continue
            now = items[clus_col].decode('gbk')
            now = extract_filename(now)
            clus[now] = 1
            #print now
            y.append(now)
            
        cnt += 1

    yy = np.array(y)
    np.save(npy, yy)

    print "length:", len(y)
    print "shape:", yy.shape
    print 'clus_cnt',len(clus)
    print ''

if __name__ == '__main__':
    #将LSH点和索引转换为npy文件，作为训练集
    extract_data_from_txt(DATASET+'/LSHVector.txt',DATASET+"/train_xdata.npy")
    extract_label_from_txt(DATASET+'/LSHIndex.txt',DATASET+"/train_ylabels_song.npy",1)

    #将线性伸缩后的查询LSH点转换为npy文件，作为查询集
    extract_data_from_txt(DATASET+'/QueryLSHLSVector.txt',DATASET+"/query_xdata.npy")
    extract_label_from_txt(DATASET+'/QueryLSHLSIndex.txt',DATASET+"/query_ylabels_song.npy",0)

    #复制文件
    # shutil.copyfile(DATASET+"/train_xdata.npy",DATASET+"/test_xdata.npy")
    # shutil.copyfile(DATASET+"/train_ylabels.npy",DATASET+"/test_ylabels.npy")
    #extract_from_txt('NLSHVector.txt','NLSHVector.npy')
    print 'Done'
