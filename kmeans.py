#-*- coding:utf-8 -*-   #允许文档中有中文
"""
用k-means对训练数据聚类

聚类流程：
读入训练数据（音高序列）train_xdata.npy，聚类数手动设定为音频文件数（歌曲数），迭代数设为100
聚类结果存储在train_ylabels_kmeans.npy

观察结果：
读入每个训练数据（音高序列）所属的音频文件名（歌曲）train_ylabels_song.npy，观察聚类结果，包括：
1.类分布是否均匀（每类包含数据数目差别不大）
2.每个类中包含一些来自相同歌曲的旋律（同一歌曲的相似旋律被聚到一类）
3.每首歌的旋律分属几个类

初步观察结果基本符合上述要求

画图呈现结果：
调用MiniPlotTool.py画图工具，画出每个类的所有音高序列
类中心的音高序列用不同颜色，观察聚类是否紧密
"""

import theano.tensor as T
from scipy.cluster.vq import *
from global_para import *

def prep_data(dataset): #从npy文件读入数据
    try:
        train_x = np.load(dataset + "/train_xdata.npy")
        train_y = np.load(dataset + "/train_ylabels_song.npy")

    except:
        print >> sys.stderr, "you need the .npy python arrays"
        print >> sys.stderr, "you can produce them with txt_to_numpy.py"
        print >> sys.stderr, dataset + "/train_xdata.npy"
        print >> sys.stderr, dataset + "/train_ylabels_song.npy"
        sys.exit(-1)

    print "train_x shape:", train_x.shape
    print "train_y shape:", train_y.shape

    return [train_x,train_y]

def write_npy(data,npy):  #将聚类结果写入文件
    y = []
    for item in data:
        y.append(item)

    yy = np.array(y)
    np.save(npy, yy)

    print "length:", len(y)
    print "shape:", yy.shape
    print ''

def cal_vector_dis(vec1,vec2):  #计算两向量欧氏距离
    if len(vec1)!=len(vec2):
        print >> sys.stderr, "two vector do not have same length"
        sys.exit(-1)

    dis = 0
    for i in range(0,len(vec1)):
        dis += pow(vec1[i]-vec2[i],2)
    return dis

def cal_cluster_cost(obs,centroid,label): #准则函数，计算所有聚类内部的均方误差和
    cost = 0
    for i in range(0,len(obs)):
        cost += cal_vector_dis(obs[i],centroid[label[i]])
    return cost

def K_means(dataset,k,iter=30):  #将dataset聚为k类iter为迭代次数
    train_x,train_y = prep_data(dataset)
    #train_x = whiten(train_x)

    #k-means聚类，centroid为聚类中心值，label为聚类结果（每个数据的聚类号）
    centroid,label = kmeans2(train_x,k,iter)

    print 'cost:',cal_cluster_cost(train_x,centroid,label)

    from collections import Counter
    c = Counter(label)    #统计label的每个值的个数，即每个分类的类名和属于该分类的样本数
    print c

    # lis = []
    # for i in range(0,len(label)):   #打印每个采样点所属的类和对应的文件名（以文件名为序））
    #     lis.append((label[i],train_y[i],i)) #存储采样点的类号，文件名，序号
    
    # lis.sort()  #二维数据sort按照第一个元素排序
    # for i,j,k in lis: #打印每个采样点所属的类和对应的文件名（以类名为序））
    #     print i,j

    #将聚类结果写入文件
    write_npy(label,dataset + "/train_ylabels_kmeans.npy")
    return [centroid,label]


def draw_cluser(dataset,centroid,label,k):  #画出前k个聚类的曲线
                                            #centroid为聚类中心值，label为聚类结果（每个数据的聚类号）
    if k<1:
        print >> sys.stderr, "you need print at least 1 cluster"
        sys.exit(-1)

    train_x,train_y = prep_data(dataset)

    lis = []
    for i in range(0,len(label)):
        lis.append((label[i],i)) #存储采样点的类号，序号
    lis.sort()  #二维数据sort按照第一个元素排序

    from collections import Counter
    c = Counter(label)    #统计label的每个值的个数，即每个分类的类名和属于该分类的样本数

    from MiniPlotTool import *
    x_value = [i for i in range(0,len(train_x[0]))] #画图的x值，0-len，len为采样点维数
    colors = ['red','green','yellow','blue','black','cyan','magenta']   #每个聚类的颜色选择

    prei = lis[0][0]    #前一个聚类号
    clus_cnt = 0    #已经打印的聚类数

    baseConfig = {
        'grid' : True,
        'title': 'Cluster No. '+str(prei)+' has '+str(c[prei])+' sample(s)'
    }
    tool = MiniPlotTool(baseConfig) #初始化图

    for i,j in lis:
        if i != prei:   #遍历到下一个聚类，打印上一个聚类
            clus_cnt += 1

            lineConf = {
                'X' : x_value,
                'Y' : centroid[prei],
                'linewidth' : 3,
                'color': colors[(prei+1)%len(colors)]
            }
            tool.addline(lineConf)  #最后添加聚类中心
            tool.plot()
            tool.show() #打印图
            if clus_cnt>=k:
                break

            baseConfig = {
                'grid' : True,
                'title': 'Cluster No. '+str(i)+' has '+str(c[i])+' sample(s)'
            }
            tool = MiniPlotTool(baseConfig) #初始化下一个类的图
            prei = i

        lineConf = {
            'X' : x_value,
            'Y' : train_x[j],    #y值为采样点值
            'color': colors[i%len(colors)]
        }
        
        # print 'Cluster No.',i,'has sample No.',j
        # print 'Line id',tool.addline(lineConf)    #添加线
        tool.addline(lineConf)    #添加线

    if clus_cnt < k:    #数据中的类数小于等于k，最后一个类会漏掉，打印最后一个
        lineConf = {
            'X' : x_value,
            'Y' : centroid[prei],
            'linewidth' : 3,
            'color': colors[(prei+1)%len(colors)]
        }
        tool.addline(lineConf)  #最后添加聚类中心
        tool.plot()
        tool.show() #打印图

if __name__ == '__main__':
    centroid,label = K_means(DATASET,K,100) #对DATASET数据聚类，聚类数K，迭代数100
    draw_cluser(DATASET,centroid,label,10)  #画出前k个聚类的曲线
                                            #centroid为聚类中心值，label为聚类结果（每个数据的聚类号）
    print 'Done'