#-*- coding:utf-8 -*-   #允许文档中有中文
"""
建立DBN网络，进行查询，输出结果

查询文件的数据维数必须与DBN数据维数相同
"""

import functools
import sys,math
import os
sys.path.append(os.getcwd())
from collections import Counter
from DBN import DBN
from global_para import *

usage = """
python result.py OUTPUT[.txt] INPUT_QUERY
        [--d DBN_PICKLED_FILE]

Exclusive uses of these options:
    --d followed by a pickled DBN file
"""


def compute_likelihoods_dbn(dbn, mat, depth=np.iinfo(int).max, normalize=True, unit=False):
    """compute the log-likelihoods of each states i according to the Deep 
    Belief Network (stacked RBMs) in dbn, for each line of mat (input data) 
    depth is the depth of the DBN at which the likelihoods will pop out,
    if None, the full DBN is used"""
    """根据给出的dbn模型，计算每个状态i的对数似然性
    对输入数据mat的每一行，在DBN网络深度为depth时返回似然性
    若depth设置为空使用整个DBN网络
    """
    # first normalize or put in the unit ([0-1]) interval
    # TODO do that only if we did not do that at the full scale of the corpus
    # 查询数据预处理，注意要和prep_qbh.py中对训练数据的处理方法一致
    if normalize:
        # if the first layer of the DBN is a Gaussian RBM, we need to normalize mat
        mat = (mat - np.mean(mat, 0)) / np.std(mat, 0)
    elif unit:
        # if the first layer of the DBN is a binary RBM, send mat in [0-1] range
        mat = (mat - np.min(mat, 0)) / np.max(mat, 0)

    import theano.tensor as T
    ret = np.ndarray((mat.shape[0], dbn.logLayer.b.shape[0].eval()), dtype=X_DTYPE)
    from theano import shared#, scan
    # propagating through the deep belief net
    batch_size = mat.shape[0] / N_BATCHES_DATASET   #计算有多少组小批量数据
    max_layer = dbn.n_layers
    out_ret = None
    if depth < dbn.n_layers:
        max_layer = depth
        print 'max layer reduce to',max_layer
        out_ret = np.ndarray((mat.shape[0], dbn.rbm_layers[max_layer].W.shape[1].eval()), dtype=X_DTYPE)
    else:
        out_ret = np.ndarray((mat.shape[0], dbn.logLayer.b.shape[0].eval()), dtype=X_DTYPE)

    #遍历每个小批量数据
    for ind in xrange(0, mat.shape[0]+1, batch_size):
        output = shared(mat[ind:ind+batch_size])    #取当前小批量数据
        print "evaluating the DBN on all the test input"

        #遍历DBN每一层
        for layer_ind in xrange(max_layer):
            [pre, output] = dbn.rbm_layers[layer_ind].propup(output)
        if depth >= dbn.n_layers:
            print "dbn output shape", output.shape.eval()
            ret = T.nnet.softmax(T.dot(output, dbn.logLayer.W) + dbn.logLayer.b)
            out_ret[ind:ind+batch_size] = T.log(ret).eval()
        else:
            out_ret[ind:ind+batch_size] = T.log(output).eval()
    return out_ret


def process(ofname, iqueryfname, idbnfname):
    '''处理函数，读取DBN网络，进行查询，输出结果
    ofname: 输出文件
    iqueryfname: 查询文件，每行一个查询，维数必须为N_FRAMES
    idbnfname: DBN模型
    '''

    #建立DBN网络模型
    dbn = None
    if idbnfname != None:
        with open(idbnfname,'r') as idbnf:
            dbn = cPickle.load(idbnf)
        #partial函数：先将不变的参数dbn导入函数，防止每次调用函数重复导入
        likelihoods_computer = functools.partial(compute_likelihoods_dbn, dbn)
        # like that = for GRBM first layer (normalize=True, unit=False)
        # TODO correct the normalize/unit to work on full test dataset

    if dbn != None:
        input_n_frames = dbn.rbm_layers[0].n_visible 
        print "this is a DBN with", input_n_frames, "frames on the input layer"
        try: 
            print "loading query from pickled file", iqueryfname
            #读取查询
            query = np.load(iqueryfname)
        except:
            #读取失败，初始化为0
            query = np.ndarray((0, N_FRAMES), dtype=X_DTYPE)
            print 'Fail to read query file:',iqueryfname

    print 'query shape:',query.shape

    #计算似然性（查询属于每个类的概率）
    print "computing likelihoods"
    likelihoods = None
    if dbn != None:
        likelihoods = likelihoods_computer(query)
        #mean_dbns = np.mean(tmp_likelihoods, 0)
        #print 'likelihoods',likelihoods
        print 'likelihoods shape',likelihoods.shape
        #print likelihoods[0]
        #print likelihoods[0].shape

    #存储结果
    ans = np.ndarray((0, likelihoods.shape[1]), dtype=Y_DTYPE)
    for likelihood in likelihoods:
        lis = [(likelihood[i],i) for i in range(0,len(likelihood))]
        lis.sort()  
        lis.reverse()   #按概率从大到小排序
        tmp = np.array([[lis[i][1] for i in range(0,len(lis))]])
        tmp = tmp.astype(int)
        ans = np.append(ans, tmp, axis=0)   #设置axis，否则会变成一维数组

    #print 'ans',ans
    print 'ans shape',ans.shape
    #保存查询结果
    np.save(ofname,ans)

def show_result(query_xdata_fname = DATASET+"/query_xdata.npy",
                output_fname = DATASET+'/query_result.npy',
                query_ylabels_song_fname = DATASET+"/query_ylabels_song.npy",
                train_xdata_fname = DATASET+"/train_xdata.npy",
                train_ylabels_song_fname = DATASET+"/train_ylabels_song.npy",
                train_ylabels_kmeans_fname = DATASET+"/train_ylabels_kmeans.npy",
                to_int_and_to_state_dicts_fname = DATASET+'/to_int_and_to_state_dicts_tuple.pickle',
                candidate_size = 100, isDraw = True,drawnum=10):
    '''
    query_xdata_fname: 查询文件，每行一个查询，维数必须为N_FRAMES，此处为经过LS之后的DBN查询数据，为按帧抽取的音高序列
    output_fname: 输出文件，存储DBN分类结果，每行为按似然性从大到小对所属类排序
    query_ylabels_song_fname: 查询的正确结果，存储每个查询所属的音频文件名
    train_xdata_fname: DBN训练数据，每行存储一个音高序列
    train_ylabels_song_fname: DBN训练数据标签，存储每个数据所属的音频文件名
    train_ylabels_kmeans_fname: DBN训练数据标签，存储每个数据所属的聚类号
    to_int_and_to_state_dicts_fname: 存储两个字典，记录标签的类名（此处为k-means聚类号）和类序号映射
    candidate_size: 每个查询的候选集大小
    isDraw: 是否打印查询数据和对应的正确候选的训练数据
    drawnum: 打印的查询数据数
    '''

    candidate = []      #存储每个查询的候选
    correct_candidate = []  #存储每个查询的正确候选（和查询属于同一首歌的候选）

    #to_int记录{类名,类序号}map,to_state记录{类序号，类名}map
    #此处类序号就是dbn_clus_result中存储的序号，类名就是k-means聚类号
    with open(to_int_and_to_state_dicts_fname) as titsf:
        to_int,to_state = cPickle.load(titsf)

    #print 'to_int',to_int

    clus_to_xdata = []   #存储每个聚类包含的训练数据序号

    #读入训练数据所属聚类号
    train_ylabels_kmeans = np.load(train_ylabels_kmeans_fname) 
    c_train_ylabels_kmeans = Counter(train_ylabels_kmeans)  #存储训练数据中每个聚类包含的数据数

    #开辟空间，列表每个元素为一个空子列表，用于存储这个聚类包含的训练数据序号
    for i in range(0,len(c_train_ylabels_kmeans)):  
        clus_to_xdata.append([])

    for i, e in enumerate(train_ylabels_kmeans):
        re = to_int[e]
        train_ylabels_kmeans[i] = re    #将类名（k-means聚类号）转换为类序号
        clus_to_xdata[re].append(i)     #记录每个聚类包含的训练数据序号

    #print 'clus_to_xdata',clus_to_xdata
    print 'number in each cluster',[[i,len(clus_to_xdata[i])] for i in range(0,len(clus_to_xdata))]

    #读入训练数据所在的音频文件名（歌曲名）
    train_ylabels_song = np.load(train_ylabels_song_fname)
    c_train_ylabels_song = Counter(train_ylabels_song)  #存储训练数据中每首歌包含的数据数
    print 'train data:',len(train_ylabels_song),'samples from',len(c_train_ylabels_song),'songs'

    #读入查询数据所在的音频文件名（歌曲名）
    query_ylabels_song = np.load(query_ylabels_song_fname)
    c_query_ylabels_song = Counter(query_ylabels_song)  #存储查询数据中每首歌包含的数据数
    print 'query data:',len(query_ylabels_song),'samples from',len(c_query_ylabels_song),'songs'

    #读DBN分类结果
    output = np.load(output_fname)

    print 'query no. | candidate(correct/total) | correct song | correct sampls in train set'

    has_correct_num = 0     #记录有正确候选的查询数

    for qi in range(0,len(output)):
        items = output[qi]  #当前查询的分类结果
        candidate.append([])    #开辟一个空列表，存储当前查询的候选集

        for i in items:
            candidate[qi].extend(clus_to_xdata[i])   #按顺序将聚类包含的数据序号加入候选集
            if len(candidate[qi]) >= candidate_size: #候选集大小满足要求
                break

        correct_candidate.append([])    #开辟一个空列表，存储当前查询的正确候选集（属于同一首歌）
        correct_songname = query_ylabels_song[qi]   #当前查询所属的音频文件名（正确歌曲名）

        for i in candidate[qi]:     #遍历候选集
            if train_ylabels_song[i] == correct_songname:  #候选所属歌曲和正确歌曲匹配
                correct_candidate[qi].append(i)  #加入正确候选集

        if len(correct_candidate[qi])>0:
            has_correct_num += 1

        #正确候选在候选集中的比例,正确歌曲名以及其在训练集中的样本数
        print qi,len(correct_candidate[qi]),'/',len(candidate[qi]),correct_songname,c_train_ylabels_song[correct_songname]

    print has_correct_num,'/',len(candidate),'query(s) has correct candidates'
    #打印查询数据和对应的正确候选的训练数据的音高曲线
    if isDraw:
        from MiniPlotTool import *
        train_xdata = np.load(train_xdata_fname)    #读入训练数据
        query_xdata = np.load(query_xdata_fname)    #读入查询数据

        x_value = [i for i in range(0,len(train_xdata[0]))] #画图的x值，0-len，len为采样点维数
        colors = ['red','green','yellow','blue','black','cyan','magenta']   #每个聚类的颜色选择

        draw_cnt = 0    #记录已经打印的查询数
        for qi in range(0,len(query_xdata)):

            if draw_cnt >= drawnum:
                break

            if(len(correct_candidate[qi]) > 0):
                baseConfig = {
                    'grid' : True,
                    'title': 'Query No.'+str(qi)+' has '+str(len(correct_candidate[qi]))+' correct candidate(s)'
                }
                tool = MiniPlotTool(baseConfig) #初始化图
                lineConf = {
                    'X' : x_value,
                    'Y' : query_xdata[qi],
                    'linewidth' : 3,
                    'color': colors[(qi+1)%len(colors)]
                }
                tool.addline(lineConf)  #添加查询的音高曲线

                for i in correct_candidate[qi]:
                    lineConf = {
                        'X' : x_value,
                        'Y' : train_xdata[i],
                        'color': colors[qi%len(colors)]
                    }
                    tool.addline(lineConf)  #添加正确歌曲对应训练数据的音高曲线
                tool.plot()
                tool.show() #打印图

                draw_cnt += 1


if __name__ == "__main__":

    dbn_fname = DBN_PICKLED_FILE    #DBN pickle文件的路径
    print "will use the following DBN to estimate states likelihoods", dbn_fname
    
    output_fname = DATASET+'/query_result.npy'   #输出文件，存储DBN分类结果
    query_xdata_fname = DATASET+"/query_xdata.npy"  #查询文件，每行一个查询，维数必须为N_FRAMES
                                                    #此处为经过LS之后的DBN查询数据，为按帧抽取的音高序列

    #读取DBN网络，进行查询，输出结果（按似然性从大到小对所属类排序）
    process(output_fname, query_xdata_fname, dbn_fname)

    show_result()

    print 'Done'
    #控制台
    # if len(sys.argv) > 3:
    #     if '--help' in sys.argv:
    #         print usage
    #         sys.exit(0)
    #     args = dict(enumerate(sys.argv))
    #     options = filter(lambda (ind, x): '--' in x[0:2], enumerate(sys.argv))
    #     dbn_fname = None # DBN cPickle
    #     if len(options): # we have options
    #         for ind, option in options:
    #             args.pop(ind)
    #             if option == '--d':
    #                 dbn_fname = args[ind+1]
    #                 args.pop(ind+1)
    #                 print "will use the following DBN to estimate states likelihoods", dbn_fname
    #     else:
    #         print "need to load DBN."
    #         sys.exit(-1)
    #     output_fname = args.values()[1]
    #     input_query_fname = args.values()[2]
    #     process(output_fname, input_query_fname, dbn_fname)
    # else:
    #     print usage
    #     sys.exit(-1)
