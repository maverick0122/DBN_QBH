#-*- coding:utf-8 -*-   #允许文档中有中文
import theano, copy, sys, json, cPickle
import theano.tensor as T
import numpy as np

BORROW = True # True makes it faster with the GPU
USE_CACHING = False # beware if you use RBM / GRBM or gammatones / speaker labels alternatively, set it to False


def prep_data(dataset, scaling='normalize',
        pca_whiten=0):
    xname = "xdata"
    try:
        train_x = np.load(dataset + "/aligned_train_xdata.npy")
        train_y = np.load(dataset + "/aligned_train_ylabels.npy")

    except:
        print >> sys.stderr, "you need the .npy python arrays"
        print >> sys.stderr, "you can produce them with txt_to_numpy.py"
        print >> sys.stderr, dataset + "/aligned_train_xdata.npy"
        print >> sys.stderr, dataset + "/aligned_train_ylabels.npy"
        sys.exit(-1)

    print "train_x shape:", train_x.shape

    if scaling == 'unit':
        ### Putting values on [0-1]
        train_x = (train_x - np.min(train_x, 0)) / np.max(train_x, 0)
    elif scaling == 'normalize':
        ### Normalizing (0 mean, 1 variance)
        # TODO or do that globally on all data (but that would mean to know
        # the test set and this is cheating!)
        train_x = (train_x - np.mean(train_x, 0)) / np.std(train_x, 0)
    elif scaling == 'student':
        ### T-statistic
        train_x = (train_x - np.mean(train_x, 0)) / np.std(train_x, ddof=1)
    if pca_whiten: 
        ### PCA whitening, beware it's sklearn's and thus stays in PCA space
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_whiten, whiten=True)
        if pca_whiten < 0:
            pca = PCA(n_components='mle', whiten=True)
        train_x = pca.fit_transform(train_x)
        with open('pca_' + xname + '.pickle', 'w') as f:
            cPickle.dump(pca, f)
    train_x_f = train_x

    ### Labels (Ys)
    from collections import Counter
    c = Counter(train_y)    #统计train_y的每个值的个数，即每个分类的类名和属于该分类的样本数
    to_int = dict([(k, c.keys().index(k)) for k in c.iterkeys()])   #to_int记录{类名,类序号}map
    to_state = dict([(c.keys().index(k), k) for k in c.iterkeys()]) #to_state记录{类序号，类名}map
    # print to_int
    # print to_state
    with open('to_int_and_to_state_dicts_tuple.pickle', 'w') as f:
        cPickle.dump((to_int, to_state), f)

    print "preparing / int mapping Ys"
    train_y_f = np.zeros(train_y.shape[0], dtype='int64')
    for i, e in enumerate(train_y):
        train_y_f[i] = to_int[e]    #记录每个样本对应的类号
        # print train_y_f[i]

    return [train_x_f, train_y_f]


def load_data(dataset, scaling='normalize', 
        pca_whiten=0, valid_cv_frac=0.2, test_cv_frac=0.2,
        numpy_array_only=False):
    """ 
    params:
     - dataset: folder
     - nframes: number of frames to replicate/pad
     - scaling: 'none' || 'unit' (put all the data into [0-1])
                || 'normalize' ((X-mean(X))/std(X))
                || student ((X-mean(X))/std(X, deg_of_liberty=1))
     - pca_whiten: not if 0, MLE if < 0, number of components if > 0
     - valid_cv_frac: valid cross validation fraction on the train set
     - test_cv_frac: test cross validation fraction on the train set
    """
    params = {'scaling': scaling,
              'pca_whiten_mfcc_path': 'pca_' + str(pca_whiten) + '.pickle' if pca_whiten else 0,
              'valid_cv_frac': valid_cv_frac,
              'test_cv_frac': test_cv_frac,
              'theano_borrow?': BORROW,
              'use_caching?': USE_CACHING}
    with open('prep_' + '_params.json', 'w') as f:
        f.write(json.dumps(params))


    def prep_and_serialize():
        [train_x, train_y] = prep_data(dataset, scaling=scaling,
                pca_whiten=pca_whiten)
        with open('train_x_' + scaling + '.npy', 'w') as f:
            np.save(f, train_x)
        with open('train_y_' + scaling + '.npy', 'w') as f:
            np.save(f, train_y)
        print ">>> Serialized all train/test tables"
        return [train_x, train_y]

    if USE_CACHING:
        try: # try to load from serialized filed, beware
            with open('train_x_' + scaling + '.npy') as f:
                train_x = np.load(f)
            with open('train_y_' + scaling + '.npy') as f:
                train_y = np.load(f)
        except: # do the whole preparation (normalization / padding)
            [train_x, train_y] = prep_and_serialize()
    else:
        [train_x, train_y] = prep_and_serialize()

    from sklearn import cross_validation
    X_train1, X_validate, y_train1, y_validate = cross_validation.train_test_split(train_x, train_y, test_size=valid_cv_frac, random_state=0)
    real_test_cv_frac = test_cv_frac / (1.0 - valid_cv_frac)
    X_train, test_x, y_train, test_y = cross_validation.train_test_split(X_train1, y_train1, test_size=real_test_cv_frac, random_state=0)
    if numpy_array_only:
        train_set_x = X_train
        train_set_y = np.asarray(y_train, dtype='int64')
        val_set_x = X_validate
        val_set_y = np.asarray(y_validate, dtype='int64')
        test_set_x = test_x
        test_set_y = np.asarray(test_y, dtype='int64')
    else:
        train_set_x = theano.shared(X_train, borrow=BORROW)
        train_set_y = theano.shared(np.asarray(y_train, dtype=theano.config.floatX), borrow=BORROW)
        train_set_y = T.cast(train_set_y, 'int64')
        val_set_x = theano.shared(X_validate, borrow=BORROW)
        val_set_y = theano.shared(np.asarray(y_validate, dtype=theano.config.floatX), borrow=BORROW)
        val_set_y = T.cast(val_set_y, 'int64')
        test_set_x = theano.shared(test_x, borrow=BORROW)
        test_set_y = theano.shared(np.asarray(test_y, dtype=theano.config.floatX), borrow=BORROW)
        test_set_y = T.cast(test_set_y, 'int64')

    return [(train_set_x, train_set_y), 
            (val_set_x, val_set_y),
            (test_set_x, test_set_y)] 
