#!/usr/bin/env python
# -*- coding=utf-8 -*-

import os
import sys
reload(sys)
import cPickle as pkl
import pandas as pd
import numpy as np
import argparse
import matplotlib
import time, datetime
import re
matplotlib.use('Agg')
import matplotlib.pyplot as plt
seed = 1234
np.random.seed(seed)

from random import shuffle
from collections import Counter
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten, Dropout, concatenate, add
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


parser = argparse.ArgumentParser(description='embedconv training')
parser.add_argument('-batchsize', dest='batchsize', type=int, default=1024, help='size of one batch')
parser.add_argument('-init', dest='init', action='store_true', default=True, help='initialize vector')
parser.add_argument('-noinit', dest='init', action='store_false', help='no initialize')
parser.add_argument('-trainable', dest='trainable', action='store_true', default=True, help='embedding vectors trainable')
parser.add_argument('-notrainable', dest='trainable', action='store_false', help='not trainable')
parser.add_argument('-transform', dest='transform', action='store_true', default=False, help='transformation of the cost')
parser.add_argument('-test', dest='test', action='store_true', default=False, help='only test step')
parser.add_argument('-filter', dest='filter', action='store_true', default=True, help='filter rare codes')
parser.add_argument('-isdays', dest='isdays', action='store_true', default=False, help='prediction of length of stay')
parser.add_argument('-dropout', dest='dpt', type=float, default=0.2, help='drop out rate')
parser.add_argument('-filtersize', dest='fz', type=int, default=3, help='filter region size')
parser.add_argument('-filternumber', dest='fn', type=int, default=200, help='filter numbers')
parser.add_argument('-lr', dest='lr', type=float, default=0.005, help='learning rate')
parser.add_argument('-maxlen', dest='maxlen', type=int, default=21, help='max sequence length')
parser.add_argument('-dim', dest='dim', type=int, default=600, help='embedding vector length')
parser.add_argument('-window', dest='window', type=int, default=10, help='word2vec window size')
args = parser.parse_args()

dropout = args.dpt
filter_size = args.fz
filter_number = args.fn
lr = args.lr
embedding_vector_length = args.dim
print 'embedding vector length %d'%(embedding_vector_length)
if args.test:
    MID = 29
    SID = 48
else:
    i = datetime.datetime.now()
    MID = i.minute
    SID = i.second
path = 'weighted1_Glove_%s:%s_%sdays_%sinit_%strainable_%fdpt_%flr_%dfz_%dfn_%dmaxlen_%ddim_%sfilter_%stransform %sbatch_%swindow'%(MID, SID, args.isdays, args.init, args.trainable, args.dpt, args.lr, args.fz, args.fn, args.maxlen, args.dim, args.filter, args.transform,args.batchsize,args.window)
f = open('./cnn_logs/%s.log'%(path),'w')
p = r'^[A-Z]'
pattern = re.compile(p)

def DataClean ():
    """ Load .csv data and do simple data cleaning
    save pd dataframe file to ./data/dataclean.df"""
    data = pd.read_csv('./data/hqmsts_1204.csv', encoding='gbk')
    data = data[['P2', 'P5', 'P7', 'P27',
        'P321', 'P324', 'P327', 'P3291', 'P3294', 'P3297', 'P3281', 'P3284', 'P3287', 'P3271', 'P3274',
        'P490', 'P4911', 'P4922', 'P4533', 'P4544', 'P45002', 'P45014', 'P45026', 'P45038', 'P45050',
        'P782']]
    #print '#Rows of raw csv: ', len(data.index)
    data = data[data['P5'].isin([1,2])]
    data = data.dropna(subset=['P321', 'P490', 'P2', 'P5', 'P7', 'P27'])
    #print '#Rows of cleaned data: ', len(data.index)
    pkl.dump(data, open('./data/dataclean.df', 'w'))
    return data

def rare_filter(seqs, remain=0.98):
    print 'filter the rare codes......'
    keys = Counter([code for item in seqs for code in item]).keys()
    values = Counter([code for item in seqs for code in item]).values()
    keys, values = zip(*sorted(zip(keys, values), key=lambda x: x[1], reverse=True))
    n_remain = np.argmin(np.abs(np.cumsum(values)/float(np.sum(values)) - remain))
    rare_keys = keys[(n_remain+1):]
    rare_freqs = values[(n_remain+1):]
    comm_keys = keys[:(n_remain+1)]
    #filteredseqs =[[str(code) for code in item if code in comm_keys] for item in seqs]
    # print 'filter seqs: {}'.format(filteredseqs[0:2])
    print '#rare codes:%d; #common codes:%d'%(len(rare_keys), len(comm_keys))
    #l_seqs = [len(item) for item in filteredseqs]
    #l_seqs = np.array(l_seqs)
    #C_seqs = Counter(l_seqs)
    #print C_seqs
    #print "the max len:{}, min:{}".format(np.max(l_seqs),np.min(l_seqs))
    #print "the 25% quarter:{}, the 75% quarter:{}, the mean:{}".format(np.percentile(l_seqs,25),np.percentile(l_seqs,75),np.mean(l_seqs))
    #pkl.dump(rarecodes, open('./data/rarecodes.df','w'))
    #pkl.dump(filteredseqs, open('./data/filteredseqs.df','w'))
    return rare_keys
def histogram(cost):
    plt.figure()
    plt.hist(cost, bins=100, normed=True)
    plt.xlabel('Total expenses')
    plt.ylabel('Occurrences')
    plt.savefig('./figs/cost_100.pdf')

def histogram1(days):
    plt.figure()
    plt.hist(days, bins=100, normed=True)
    plt.xlabel('Length of Stay')
    plt.ylabel('Occurrences')
    plt.savefig('./figs/days_100.pdf')


def ToRawList(data):
    n_samples = len(data.index)
    # demographics, P27=days
    #demographics = np.zeros((n_samples, 4))
    demographics = np.zeros((n_samples, 3))
    demographics[:, 0:1] = data[['P2']].values #times
    demographics[:, 1:2] = data[['P5']].values - 1 #gender
    demographics[:, 2:3] = data[['P7']].values  #age
    #demographics[:, 3:4] = data[['P27']].values #days

    #diseases codes
    disease = data[['P321', 'P324', 'P327', 'P3291', 'P3294', 'P3297', 'P3281', 'P3284', 'P3287', 'P3271', 'P3274']]
    disease = disease.fillna('')
    disease = disease.values
    disease = [[str(code).strip() for code in item if code != ''] for item in disease]
    print disease[:5]

    main_dis = data[['P321']]
    main_dis = main_dis.fillna('')
    main_dis = main_dis.values
    #print main_dis
    #"""
    l_disease = [len(item) for item in disease]
    l_disease = np.array(l_disease)
    C_disease = Counter(l_disease)
    print "the max len:{}, min:{}".format(np.max(l_disease),np.min(l_disease))
    print "the 25% quarter:{}, the 75% quarter:{}, the mean:{}(+-){}".format(np.percentile(l_disease,25),np.percentile(l_disease,75),np.mean(l_disease),np.std(l_disease))
    #disease = [[str(code)[0:5] for code in item if code != ''] for item in disease] #code len = 4
    #"""
    # surgeries codes
    surgery = data[['P490', 'P4911', 'P4922', 'P4533', 'P4544', 'P45002', 'P45014', 'P45026', 'P45038', 'P45050']]
    surgery = surgery.fillna('')
    surgery = surgery.values
    surgery = [['#'.join(str(code).strip().split(' ')) for code in item if code != ''] for item in surgery]#code len = 5
    surgery = [[str('0'+code) if len(code.split('.')[0])==1 else code for code in item] for item in surgery]
    print surgery[:5]
    if os.path.isfile('./data/rarecodes.df'):
        rarecodes = pkl.load(open('./data/rarecodes.df','r'))
    else:
        rarecodes = []
        raredisease = rare_filter(disease)
        rarecodes.extend(raredisease)
        raresurgery = rare_filter(surgery)
        rarecodes.extend(raresurgery)
        pkl.dump(rarecodes, open('./data/rarecodes.df','w'))
    #"""
    l_surgery = [len(item) for item in surgery]
    l_disease = np.array(l_disease)
    l_surgery = np.array(l_surgery)
    C_surgery = Counter(l_surgery)
    print "the max len:{}, min:{}".format(np.max(l_surgery),np.min(l_surgery))
    print "the 25% quarter:{}, the 75% quarter:{}, the mean:{}".format(np.percentile(l_surgery,25),np.percentile(l_surgery,75),np.mean(l_surgery))
    #"""
    seqs = data[['P321', 'P324', 'P327', 'P3291', 'P3294', 'P3297', 'P3281', 'P3284', 'P3287', 'P3271', 'P3274',
                 'P490', 'P4911', 'P4922', 'P4533', 'P4544', 'P45002', 'P45014', 'P45026', 'P45038', 'P45050']]
    seqs = seqs.fillna('')
    seqs = seqs.values
    seqs = [['#'.join(str(code).strip().split(' ')) for code in item if code != ''] for item in seqs]#replace the space with '#''
    seqs = [[str('0'+code) if len(code.split('.')[0])==1 else code for code in item] for item in seqs] #replace '3.90034' with the '03.90034'
    print seqs[:5]
    #seqs = [[str(code) for code in item if code != ''] for item in seqs]
    #l_seqs = [len(item) for item in seqs]
    #l_seqs = np.array(l_seqs)
    #C_seqs = Counter(l_seqs)
    #print C_seqs
    #print "the max len:{}, min:{}".format(np.max(l_seqs),np.min(l_seqs))
    #print "the 25% quarter:{}, the 75% quarter:{}, the mean:{}".format(np.percentile(l_seqs,25),np.percentile(l_seqs,75),np.mean(l_seqs))
    cost = data[['P782']].values
    cost = np.asarray(cost, dtype=np.float32)
    days = data[['P27']].values
    days = np.asarray(days, dtype=np.float32)
    # cost
    rm_rows = []
    thr_high = np.percentile(cost.squeeze(), 97)
    thr_low = np.percentile(cost.squeeze(), 3)
    days_thr_high = np.percentile(days.squeeze(), 97)
    days_thr_low = np.percentile(days.squeeze(), 3)
    for i in xrange(n_samples):
        if len(seqs[i]) <= 1 or cost[i,0] > thr_high or cost[i,0] < thr_low or days[i,0] > days_thr_high or days[i,0] < days_thr_low:
            rm_rows.append(i)

    main_dis = np.delete(main_dis, rm_rows, axis=0)
    demographics = np.delete(demographics, rm_rows, axis=0)
    seqs = np.delete(seqs, rm_rows, axis=0)
    disease = np.delete(disease, rm_rows, axis=0)
    surgery = np.delete(surgery, rm_rows, axis=0)
    cost = np.delete(cost, rm_rows, axis=0)
    days = np.delete(days, rm_rows, axis=0)
    n_samples = len(seqs)
    print >>f,'#samples:%d'%(n_samples)
    # cost
    print >>f,'Cost summary: The max:{}, the min:{}'.format(np.max(cost), np.min(cost))
    print >>f,"the 25% quarter:{}, the 75% quarter:{}, the mean:{}(+-){}".format(np.percentile(cost,25),np.percentile(cost,75),np.mean(cost),np.std(cost))
    histogram(cost)
    # days
    print >>f,'Days summary: The max:{}, the min:{}'.format(np.max(days), np.min(days))
    print >>f,"the 25% quarter:{}, the 75% quarter:{}, the mean:{}(+-){}".format(np.percentile(days,25),np.percentile(days,75),np.mean(days),np.std(days))
    histogram1(days)
    main_dis = [[str(code) for code in item if code != ''] for item in main_dis]
    #print main_dis
    C_maincodes = Counter([code for seq in main_dis for code in seq])
    #print C_maincodes
    main_code = C_maincodes.keys()
    n_dim = len(main_code)
    print n_dim
    code2id = dict(zip(main_code, range(n_dim)))
    maincodemat = np.zeros((n_samples, n_dim), dtype=np.float32)
    for i in xrange(n_samples):
        for code in main_dis[i]:
            if code in code2id:
                index = code2id[code]
                maincodemat[i,index] += 1
    #demographics = np.hstack((demographics, maincodemat))
    Data = (seqs, cost, days, demographics,disease, surgery, rarecodes)
    pkl.dump(Data, open('./data/Data.df','w'))
    return seqs, cost, days, demographics, disease, surgery, main_dis, rarecodes

def token_to_index(seqs):
    C_codes = Counter([code for seq in seqs for code in seq])
    code_index = {}
    for idx, item in enumerate(C_codes.keys()):
        code_index[item] = idx + 1
    #print C_codes
    #print code_index # starting from 1
    print "the #unique codes: {}".format(len(C_codes.keys()))#the unique code is 16146
    return code_index

def get_index_embedding(code_index={}, level=0):
    index_embedding = {}
    cnt = 0
    #model = Word2Vec.load('./cnn_model/newlevel%d_word2vec_dim%d_window%d.model' %(level, args.dim, args.window))
    dim = 600
    window = 21
    #model = KeyedVectors.load('./cnn_model/newlevel%d_glove_dim%d_window%d.model' %(level, dim, window))
    model = KeyedVectors.load('./cnn_model/newlevel%d_weighted0.5_glove_dim%d_window%d.model' %(level, dim, window))
    #print model.similar_by_word('E11.901')
    #print model.word_vec('E11.901')
    for code, index in code_index.items():
        #print re.findall(pattern, code)
        if len(re.findall(pattern, code))== 1:
            if level == 0:
                newcode = code
            if level == 1:
                newcode = code[0:3]
            if level == 2:
                newcode = code[0:5]
            if level == 3:
                newcode = code[0:6]
        else:
            if level == 0:
                newcode = code
            if level == 1:
                newcode = code[0:3]
            if level == 2:
                newcode = code[0:4]
            if level == 3:
                newcode = code[0:5]

        if newcode in model:
            index_embedding[index] = model[newcode]
        else:
            cnt = cnt + 1
            index_embedding[index] = np.random.uniform(-0.25,0.25,embedding_vector_length)
        #print '#{}/#{} not in training vectors'.format(cnt, len(code_index))
    return index_embedding

def get_trained_embedding(index_embedding=None):
    index_sorted = sorted(index_embedding.items()) # sorted by index starting from 1
    trained_embedding = [t[1] for t in index_sorted]
    embedding_vector_length = args.dim
    zeros = np.random.uniform(-0.25,0.25,embedding_vector_length)
    trained_embedding = np.vstack((zeros, trained_embedding))
    trained_embedding =  np.array(trained_embedding)
    print trained_embedding.shape
    return trained_embedding

def embedding_encoder(idseqs, index_embedding):
    n_samples = len(idseqs)
    dim = args.dim
    mat = np.zeros((n_samples, dim), np.float32)
    for i in xrange(n_samples):
        for codeid in idseqs[i]:
            if codeid in index_embedding:
                code_embedding = np.array(index_embedding[codeid])
            else:
                code_embedding = np.random.uniform(-0.25, 0.25, dim)
            mat[i] += code_embedding
    mat = np.array(mat)
    return mat

def multichannel_embedding_encoder(
        idseqs, index_embedding,index_embedding1,
        index_embedding2,index_embedding3):
    n_samples = len(idseqs)
    dim = args.dim
    mat = np.zeros((n_samples, dim), np.float32)
    for i in xrange(n_samples):
        for codeid in idseqs[i]:
            code_embedding = np.asarray([0]*dim, np.float32)
            if codeid in index_embedding:
                code_embedding += np.array(index_embedding[codeid])
            if codeid in index_embedding1:
                code_embedding += np.array(index_embedding1[codeid])
            if codeid in index_embedding2:
                code_embedding += np.array(index_embedding2[codeid])
            if codeid in index_embedding3:
                code_embedding += np.array(index_embedding3[codeid])
            mat[i] += code_embedding
    mat = np.array(mat)
    return mat

def one_hot_encoder(seqsind, nb_words):
    print 'one_hot encoding...'
    n_samples = len(seqsind)
    n_dim = nb_words
    mat = np.zeros((n_samples, n_dim), dtype=np.float32)
    for i in xrange(n_samples):
        for codeid in seqsind[i]:
            if codeid !=0:
                index = codeid - 1 # for the seqid starts from 1
                mat[i,index] = mat[i,index] + 1
    return mat

def svd(seqsind, dim=600):
    mat = one_hot_encoder(seqsind, nb_words)
    print mat.shape
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=dim, random_state=42)
    print 'SVD......'
    svd.fit(mat)
    res = svd.transform(mat)
    print 'SVD done!'
    return res

def glove(level, dim=600, window=21):
    #vectors = './GloVe/vectors%d'%(level)
    vectors = './GloVe/weighted1_vectors%d'%(level)
    model = KeyedVectors.load_word2vec_format(vectors, binary=False)
    name = 'newlevel%d_weighted1_glove_dim%d_window%d.model'%(level, dim, window)
    model.save('./cnn_model/'+name)
    return model
def word2vec(seqs, level, window=args.window, dim=args.dim, visualize=False):
    n_samples = len(seqs)
    sentences = []
    for sentence in seqs:
        shuffle(sentence)
        sentences.append(sentence)
    model = Word2Vec(sentences, min_count=1, window=window, size=dim)
    model.save('./cnn_model/newlevel%d_word2vec_dim%d_window%d.model' %(level, dim, window))
    print model
    if visualize:
        from tsne import tsne
        #disease_vocabulary = Counter([item for i in xrange(n_samples) for item in disease[i]]).keys()
        #surgery_vocabulary = Counter([item for i in xrange(n_samples) for item in surgery[i]]).keys()
        codes_vocabulary = Counter([code for seq in seqs for code in seq]).keys()
        X = np.vstack([model[item][np.newaxis,:] for item in codes_vocabulary])
        Y = tsne(X, 2, 200, 20.)
        #pkl.dump(disease_vocabulary, open('./logs/disease_vocabulary.pkl', 'w'))
        #pkl.dump(surgery_vocabulary, open('./logs/surgery_vocabulary.pkl', 'w'))
        pkl.dump(codes_vocabulary, open('./logs/codes_vocabulary.pkl', 'w'))
        pkl.dump(X, open('./logs/word2vec_vector.pkl', 'w'))
        pkl.dump(Y, open('./logs/word2vec_2d_vector.pkl', 'w'))
        fig, ax = plt.subplots()
        #ax.scatter(Y[:len(disease_vocabulary),0], Y[:len(disease_vocabulary),1], 20, marker='o')
        #ax.scatter(Y[-len(surgery_vocabulary):,0], Y[-len(surgery_vocabulary):,1], 20, marker='x')
        ax.scatter(Y[-len(codes_vocabulary):,0], Y[-len(codes_vocabulary):,1], 20, marker='x')
        fig.savefig('./figs1/%s_tsne.pdf'%(level))
    #return model


def load_data(X,demographics, y, onehot_mat):
    print 'Spliting train, valid, test parts...'
    indices = np.arange(n_seqs)
    np.random.shuffle(indices)
    X = X[indices]
    demographics = demographics[indices]
    y = y[indices]
    #onehot_mat = onehot_mat[indices]

    n_tr = int(n_seqs * 0.85)
    n_va = int(n_seqs * 0.05)
    n_te = n_seqs - n_tr - n_va
    X_train = X[:n_tr]
    demographics_train = demographics[:n_tr]
    y_train = y[:n_tr]
    #onehot_mat_train = onehot_mat[:n_tr]

    X_valid = X[n_tr:n_tr+n_va]
    demographics_valid = demographics[n_tr:n_tr+n_va]
    y_valid = y[n_tr:n_tr+n_va]
    #onehot_mat_valid = onehot_mat[n_tr:n_tr+n_va]

    X_test = X[-n_te:]
    demographics_test = demographics[-n_te:]
    y_test = y[-n_te:]
    #onehot_mat_test = onehot_mat[-n_te:]

    #print np.max(y_test),np.min(y_test)
    return X_train, X_test, X_valid, y_train, y_test, y_valid, demographics_train, demographics_test, demographics_valid#, onehot_mat_train,onehot_mat_test,onehot_mat_valid

def filter_test(X_test, rarecodes,index_code, threshold=2):
    print 'selecting samples with rarecodes from testing dataset.....'
    rare_inds = []
    comm_inds = []
    for ind, seq in enumerate(X_test):
        #print seq
        cnt = 0
        for index in seq:
            if index in index_code:
                code = index_code[index]
            else:
                code = 'none'
            if code in rarecodes:
                cnt = cnt + 1
        if cnt >= threshold:
            #print ind, cnt
            rare_inds.append(int(ind))
        else:
            comm_inds.append(int(ind))
    print 'rare_inds:%d, comm_inds:%d'%(len(rare_inds),len(comm_inds))#, rare_inds
    rare_inds = np.asarray(rare_inds)
    comm_inds = np.asarray(comm_inds)
    return rare_inds, comm_inds

def single_channel_merge_model(demgras_dim):
    codes_in = Input(shape=(MAX_LEN, ), dtype='float32')
    if args.init:
    	print 'initialize embedding layer with pre-training vectors'
    	print 'embedding layers trainalble %s' % args.trainable
    	embedding0level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=MAX_LEN,
                                weights=[embedding_matrix0],
                                trainable=args.trainable)(codes_in)
    else:
    	print 'one hot embedding with random initialization...'
    	embedding0level = Embedding(output_dim=embedding_vector_length,
    		                        input_dim=max_features,
    		                        embeddings_initializer='random_uniform',
    		                        input_length=MAX_LEN)(codes_in)
    conv_result = []
    for i in range(3):
        conv_layer = Conv1D(filter_number, filter_size, padding='same', activation='relu')
        conv0 = conv_layer(embedding0level)
        pooling0 = GlobalMaxPooling1D()(conv0)
        conv_result.append(pooling0)

    demgras_in = Input(shape=(demgras_dim,), dtype='float32')
    dense_demgras = Dense(3, activation='sigmoid')(demgras_in)
    conv_result.append(dense_demgras)#append the demographics information
    merge_out = concatenate(conv_result)
    dense_out = Dense(500, activation='relu')(merge_out)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(100, activation='relu')(dpt)
    dpt = Dropout(dropout)(dense_out)
    mode_out = Dense(1)(dpt)
    model = Model([codes_in, demgras_in], mode_out)
    rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['mae'])
    print(model.summary())
    return model

def single_channel_split_model(demgras_dim):
    dis_in = Input(shape=(DIS_MAX_LEN, ), dtype='float32')
    sur_in = Input(shape=(SUR_MAX_LEN, ), dtype='float32')
    if args.init:
    	print 'initialize embedding layer with pre-training vectors'
    	print 'embedding layers trainalble %s' % args.trainable
        dis_embedding0level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix0],
                                trainable=args.trainable)(dis_in)

        sur_embedding0level = Embedding(output_dim=embedding_vector_length,
                                        input_dim=max_features,
                                        input_length=SUR_MAX_LEN,
                                        weights=[embedding_matrix0],
                                        trainable=args.trainable)(sur_in)
    else:
    	print 'one hot embedding with random initialization...'
    	dis_embedding0level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                embeddings_initializer='random_uniform')(dis_in)

        sur_embedding0level = Embedding(output_dim=embedding_vector_length,
                                        input_dim=max_features,
                                        input_length=SUR_MAX_LEN,
                                        embeddings_initializer='random_uniform')(sur_in)
    conv_result = []
    for i in range(3):
        conv_layer = Conv1D(filter_number, filter_size, padding='same', activation='relu')
        conv0 = conv_layer(dis_embedding0level)
        pooling0 = GlobalMaxPooling1D()(conv0)
        conv_result.append(pooling0)

    for i in range(3):
        conv_layer = Conv1D(filter_number, filter_size, padding='same', activation='relu')
        conv0 = conv_layer(sur_embedding0level)
        pooling0 = GlobalMaxPooling1D()(conv0)
        conv_result.append(pooling0)

    demgras_in = Input(shape=(demgras_dim,), dtype='float32')
    dense_demgras = Dense(3, activation='sigmoid')(demgras_in)
    conv_result.append(dense_demgras)#append the demographics information
    merge_out = concatenate(conv_result)
    dense_out = Dense(1000, activation='relu')(merge_out)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(500, activation='relu')(dpt)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(100, activation='relu')(dpt)
    dpt = Dropout(dropout)(dense_out)
    mode_out = Dense(1)(dpt)
    model = Model([dis_in, sur_in, demgras_in], mode_out)
    rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['mae'])
    print(model.summary())
    return model

def multi_channel_split_model(demgras_dim):
    dis_in = Input(shape=(DIS_MAX_LEN, ), dtype='float32')
    dis_embedding0level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix0],
                                trainable=args.trainable)(dis_in)
    dis_embedding1level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix1],
                                trainable=args.trainable)(dis_in)
    dis_embedding2level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix2],
                                trainable=args.trainable)(dis_in)
    dis_embedding3level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=DIS_MAX_LEN,
                                weights=[embedding_matrix3],
                                trainable=args.trainable)(dis_in)
    sur_in = Input(shape=(SUR_MAX_LEN, ), dtype='float32')
    sur_embedding0level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=SUR_MAX_LEN,
                                weights=[embedding_matrix0],
                                trainable=args.trainable)(sur_in)
    sur_embedding1level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=SUR_MAX_LEN,
                                weights=[embedding_matrix1],
                                trainable=args.trainable)(sur_in)
    sur_embedding2level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=SUR_MAX_LEN,
                                weights=[embedding_matrix2],
                                trainable=args.trainable)(sur_in)
    sur_embedding3level = Embedding(output_dim=embedding_vector_length,
                                input_dim=max_features,
                                input_length=SUR_MAX_LEN,
                                weights=[embedding_matrix3],
                                trainable=args.trainable)(sur_in)
    conv_result = []
    for i in range(3):
        channel_result = []
        conv_layer = Conv1D(filter_number, filter_size, padding='same', activation='relu')
        conv0 = conv_layer(dis_embedding0level)
        conv1 = conv_layer(dis_embedding1level)
        conv2 = conv_layer(dis_embedding2level)
        conv3 = conv_layer(dis_embedding3level)
        pooling0 = GlobalMaxPooling1D()(conv0)
        pooling1 = GlobalMaxPooling1D()(conv1)
        pooling2 = GlobalMaxPooling1D()(conv2)
        pooling3 = GlobalMaxPooling1D()(conv3)
        channel_result.append(pooling0)
        channel_result.append(pooling1)
        channel_result.append(pooling2)
        channel_result.append(pooling3)
        allchannel = add(channel_result)
        conv_result.append(allchannel)

    for i in range(3):
        channel_result = []
        conv_layer = Conv1D(filter_number, filter_size, padding='same', activation='relu')
        conv0 = conv_layer(sur_embedding0level)
        conv1 = conv_layer(sur_embedding1level)
        conv2 = conv_layer(sur_embedding2level)
        conv3 = conv_layer(sur_embedding3level)
        pooling0 = GlobalMaxPooling1D()(conv0)
        pooling1 = GlobalMaxPooling1D()(conv1)
        pooling2 = GlobalMaxPooling1D()(conv2)
        pooling3 = GlobalMaxPooling1D()(conv3)
        channel_result.append(pooling0)
        channel_result.append(pooling1)
        channel_result.append(pooling2)
        channel_result.append(pooling3)
        allchannel = add(channel_result)
        conv_result.append(allchannel)

    demgras_in = Input(shape=(demgras_dim,), dtype='float32')
    dense_demgras = Dense(1000, activation='sigmoid')(demgras_in)
    conv_result.append(dense_demgras)#append the demographics information
    merge_out = concatenate(conv_result)
    dense_out = Dense(1000, activation='relu', name='F1')(merge_out)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(500, activation='relu', name='F2')(dpt)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(100, activation='relu', name='F3')(dpt)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(50, activation='relu', name='F4')(dpt)
    dpt = Dropout(dropout)(dense_out)
    dense_out = Dense(2, activation='relu', name='F5')(dpt)
    dpt = Dropout(dropout)(dense_out)
    mode_out = Dense(1)(dpt)
    model = Model([dis_in, sur_in, demgras_in], mode_out)
    rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=10-8, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['mae'])
    print(model.summary())
    return model


def cross_validation():
    estimator = KerasRegressor(build_fn=cnn_model, epochs=100, batch_size=args.batchsize, verbose=0)
    kfold = KFold(n_splits=5, random_state=seed)
    res = cross_val_score(estimator, X, y, cv=kfold)
    print 'cross validation for cnn model ......'
    print 'cnn_model: MSE %.2f (+-)%.2f'%(res.mean(), res.std())
#cross_validation()

def evaluation(y_test, y_pred):
    #print y_test[0:6],y_pred[0:6]
    #print y_test[0:6],y_pred[0:6]
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print >>f,'MSE:{}, RMSE:{}, MAE:{}, R2:{}'.format(mse, rmse, mae,r2)
    print 'MSE:{}, RMSE:{}, MAE:{}, R2:{}'.format(mse, rmse, mae,r2)
    return r2, rmse

def daysplots(y_test, y_pred, r2, rmse, name):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 35)
    #ax.plot(y_test.squeeze(), y_test.squeeze(), s=10, marker='.',c='r')
    ax.scatter(y_test.squeeze(), y_pred, s=10,marker='.', c='b')
    ax.text(28,30,'R2:{:.4f}'.format(r2))
    ax.text(28,32,'RMSE:{:.2f}'.format(rmse))
    ax.set_xlabel('True length of stay (days)')
    ax.set_ylabel('predicted length of stay (days)')
    fig.savefig('./figs/%s_days.pdf'%(name))

def costplots(y_test, y_pred, r2, rmse, name):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 120000)
    ax.set_ylim(0, 120000)
    #ax.plot(y_test.squeeze(), y_test.squeeze(), s=10, marker='.',c='r')
    ax.scatter(y_test.squeeze(), y_pred, s=10,marker='.', c='b')
    ax.text(80000,100000,'R2:{:.4f}'.format(r2))
    ax.text(80000,110000,'RMSE:{:.2f}'.format(rmse))
    ax.set_xlabel('True hospital charge (yuan)')
    ax.set_ylabel('predicted hospital charge(yuan)')
    fig.savefig('./figs/%s_cost.pdf'%(name))


#orig_stdout = sys.stdout
#modelpath = '1bestmodel_%sinit_%strainable_%fdpt_%flr_%dfz_%dfn_%dmaxlen_%ddim'%(args.init, args.trainable, args.dpt, args.lr, args.fz, args.fn, args.maxlen, args.dim)
#f = open('./cnn_logs/%s.log'%(modelpath),'w')
#sys.stdout = f

if os.path.isfile('./data/dataclean.df'):
    data = pkl.load(open('./data/dataclean.df','r'))
else:
    data = DataClean()
rawseqs, cost, days, demographics, disease, surgery, main_dis, rarecodes = ToRawList(data)

seqs1levels = []
seqs2levels = []
seqs3levels = []
for seqs in rawseqs:
    levels1 = []
    levels2 = []
    levels3 = []
    for code in seqs:
        levels1.append(code[0:3])
        #print re.findall(pattern, code)
        if len(re.findall(pattern, code))== 1:
            levels2.append(code[0:5])
            levels3.append(code[0:6])
        else:
            levels2.append(code[0:4])
            levels3.append(code[0:5])
    seqs1levels.append(levels1)
    seqs2levels.append(levels2)
    seqs3levels.append(levels3)
"""
word2vec(rawseqs, 0)
word2vec(seqs1levels, 1)
word2vec(seqs2levels, 2)
word2vec(seqs3levels, 3)

glove vectors
raw_corpus = '\n'.join([' '.join(seq) for seq in rawseqs])
f1 =  open('./GloVe/corpus', 'w')
f1.write(raw_corpus)
f1.close()
corpus1 = '\n'.join([' '.join(seq) for seq in seqs1levels])
f1 = open('./GloVe/corpus1', 'w')
f1.write(corpus1)
f1.close()
corpus2 = '\n'.join([' '.join(seq) for seq in seqs2levels])
f1 = open('./GloVe/corpus2', 'w')
f1.write(corpus2)
f1.close()
corpus3 = '\n'.join([' '.join(seq) for seq in seqs3levels])
f1 = open('./GloVe/corpus3', 'w')
f1.write(corpus3)
f1.close()
glove(0)
glove(1)
glove(2)
glove(3)
"""

code_index = token_to_index(rawseqs)# transform code into index
index_code = dict([(kv[1], kv[0]) for kv in code_index.items()])# validate the code_index
#print [[index_code[index] for index in item ] for item in seqs[0:2]]

idseqs = [[code_index[code] for code in item] for item in rawseqs]
iddis = [[code_index[code] for code in item] for item in disease]
idsur = [[code_index[code] for code in item] for item in surgery]
idmain_dis = [[code_index[code] for code in item] for item in main_dis]
#"""
index_embedding = get_index_embedding(code_index, 0)
embedding_matrix0 = get_trained_embedding(index_embedding)
index_embedding1 = get_index_embedding(code_index, 1)
embedding_matrix1 = get_trained_embedding(index_embedding1)
index_embedding2 = get_index_embedding(code_index, 2)
embedding_matrix2 = get_trained_embedding(index_embedding2)
index_embedding3 = get_index_embedding(code_index, 3)
embedding_matrix3 = get_trained_embedding(index_embedding3)

#"""
main_code = []
for item in idmain_dis:
    for ind in item:
        main_code.append(embedding_matrix0[ind])
maincodemat = np.array(main_code)
print maincodemat, maincodemat.shape

nb_words = len(index_code) # code_index starting from 1,
max_features = nb_words + 1
n_seqs = len(idseqs)
print n_seqs
MAX_LEN = args.maxlen # the max len is 21
seqs = pad_sequences(idseqs, maxlen=MAX_LEN)
DIS_MAX_LEN = 11
disease = pad_sequences(iddis, maxlen=DIS_MAX_LEN)
SUR_MAX_LEN = 10
surgery = pad_sequences(idsur, maxlen=SUR_MAX_LEN)

X = np.array(seqs)
X1 = np.array(disease)
X2 = np.array(surgery)
def extract_patientvec(model,modelpath, disease, surgery, demographics):
    model.load_weights("./cnn_model/%s.hdf5"%(modelpath),by_name=False)
    sub_model1 = Model(input=model.inputs, output=model.get_layer('F1').output)
    sub_model2 = Model(input=model.inputs, output=model.get_layer('F2').output)
    sub_model3 = Model(input=model.inputs, output=model.get_layer('F3').output)
    sub_model4 = Model(input=model.inputs, output=model.get_layer('F4').output)
    sub_model5 = Model(input=model.inputs, output=model.get_layer('F5').output)

    patientvecs3 = sub_model3.predict([disease, surgery, demographics], verbose=1)
    modelname='sub_model3'
    print patientvecs3.shape
    pkl.dump(patientvecs3, open('./cnn_model/patientvec_%s'%(modelname),'w'))

    patientvecs4 = sub_model4.predict([disease, surgery, demographics], verbose=1)
    modelname='sub_model4'
    print patientvecs4.shape
    pkl.dump(patientvecs4, open('./cnn_model/patientvec_%s'%(modelname),'w'))

    patientvecs5 = sub_model5.predict([disease, surgery, demographics], verbose=1)
    modelname='sub_model5'
    print patientvecs5.shape
    pkl.dump(patientvecs5, open('./cnn_model/patientvec_%s'%(modelname),'w'))
"""
if args.test:
    modelpath = 'MG_merge' + path
    demgras_dim = 3
    model= multi_channel_split_model(demgras_dim)
    extract_patientvec(model, modelpath, disease, surgery, demographics)
"""
#demographics = np.hstack((demographics, maincodemat))
if args.isdays:
    y = np.asarray(days, dtype='float32')
else:
    y = np.asarray(cost,dtype='float32')
if args.transform:
    y = np.log(1.0 + y)
print X, demographics

rare_inds, comm_inds= filter_test(X, rarecodes, index_code, 4)
rare_X = X[rare_inds]
comm_X = X[comm_inds]
rare_X1 = X1[rare_inds]
comm_X1 = X1[comm_inds]
rare_X2 = X2[rare_inds]
comm_X2 = X2[comm_inds]
rare_demographics = demographics[rare_inds]
comm_demographics = demographics[comm_inds]
rare_y = y[rare_inds]
comm_y = y[comm_inds]


print 'Spliting train, test parts...'
X_train, X_test, X1_train, X1_test, X2_train, X2_test,y_train, y_test, demographics_train, demographics_test = train_test_split(comm_X, comm_X1, comm_X2,comm_y, comm_demographics, test_size=0.1, random_state=seed)
# filter the samples having rare codes from test dataset
rare_inds,_ = filter_test(X_test, rarecodes,index_code, 2)
rare_X_test = X_test[rare_inds]
rare_X1_test = X1_test[rare_inds]
rare_X2_test = X2_test[rare_inds]
rare_demographics_test = demographics_test[rare_inds]
rare_y_test = y_test[rare_inds]

def onehot_RF(X_train, X_test, y_train,y_test, demographics_train, demographics_test, rare_X_test, rare_demographics_test,rare_y_test):
    #one_hot_encoder
    print 'RandomForestRegressor with one_hot encoding...'
    onehot_train = one_hot_encoder(X_train, nb_words)
    #train_mat = onehot_train
    train_mat = np.hstack((onehot_train, demographics_train))#merge the demographics info
    rf = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=-1)
    rf.fit(train_mat, y_train.ravel())
    #test_mat = onehot_test
    onehot_test = one_hot_encoder(X_test, nb_words)
    test_mat = np.hstack((onehot_test, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse =  evaluation(y_test, y_pred)

    print 'testing on filter samples...'
    rare_test = one_hot_encoder(rare_X_test, nb_words)
    rare_test_mat = np.hstack((rare_test, rare_demographics_test))#merge the demographics info
    rare_y_pred = rf.predict(rare_test_mat)
    r2, rmse = evaluation(rare_y_test, rare_y_pred)

def embedding_RF(X_train, X_test,y_train,y_test,demographics_train,demographics_test,rare_X_test,rare_demographics_test,rare_y_test, rare_X, rare_demographics, rare_y):
    #word2vec encoder
    print >>f,'RandomForestRegressor with embedding encoding...'
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info

    rf = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=-1)
    rf.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    index = 1
    name = 'SG_RF%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

    print >>f,'testing on filter testing samples...'
    rare_test_mat = embedding_encoder(rare_X_test, index_embedding)
    rare_test_mat = np.hstack((rare_test_mat, rare_demographics_test))#merge the demographics info
    rare_y_pred = rf.predict(rare_test_mat)
    r2, rmse = evaluation(rare_y_test, rare_y_pred)
    index = 2
    name = 'SG_RF%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)


    print >>f,'testing on filter samples...'
    rare_mat = embedding_encoder(rare_X, index_embedding)
    rare_mat = np.hstack((rare_mat, rare_demographics))#merge the demographics info
    rare_y_pred = rf.predict(rare_mat)
    r2, rmse = evaluation(rare_y, rare_y_pred)
    index = 3
    name = 'SG_RF%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

    #multichannel word2vec embedding encoder
    print >>f,'RandomForestRegressor with multichannel embedding encoding...'
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info

    rf = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=-1)
    rf.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = rf.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)
    index = 1
    name = 'MG_RF%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

    print >>f,'testing on filter testing samples...'
    rare_test_mat = multichannel_embedding_encoder(rare_X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    rare_test_mat = np.hstack((rare_test_mat, rare_demographics_test))#merge the demographics info
    rare_y_pred = rf.predict(rare_test_mat)
    r2, rmse = evaluation(rare_y_test, rare_y_pred)
    index = 2
    name = 'MG_RF%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

    print >>f,'testing on filter samples...'
    rare_mat = multichannel_embedding_encoder(rare_X, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    rare_mat = np.hstack((rare_mat, rare_demographics))#merge the demographics info
    rare_y_pred = rf.predict(rare_mat)
    r2, rmse = evaluation(rare_y, rare_y_pred)
    index = 3
    name = 'MG_RF%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

def embedding_LR(X_train, X_test,y_train,y_test,demographics_train,demographics_test,rare_X_test,rare_demographics_test,rare_y_test, rare_X, rare_demographics, rare_y):
    #word2vec encoder
    print >>f,'Linear regression with embedding encoding...'
    train_mat = embedding_encoder(X_train, index_embedding)
    test_mat = embedding_encoder(X_test, index_embedding)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info

    lr = LinearRegression(normalize=True, n_jobs=-1)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)

    print >>f,'testing on filter testing samples...'
    rare_test_mat = embedding_encoder(rare_X_test, index_embedding)
    rare_test_mat = np.hstack((rare_test_mat, rare_demographics_test))#merge the demographics info
    rare_y_pred = lr.predict(rare_test_mat)
    r2, rmse = evaluation(rare_y_test, rare_y_pred)

    print >>f,'testing on filter samples...'
    rare_mat = embedding_encoder(rare_X, index_embedding)
    rare_mat = np.hstack((rare_mat, rare_demographics))#merge the demographics info
    rare_y_pred = lr.predict(rare_mat)
    r2, rmse = evaluation(rare_y, rare_y_pred)

    #multichannel word2vec embedding encoder
    print >>f,'Linear regression with multichannel embedding encoding...'
    train_mat = multichannel_embedding_encoder(X_train, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    test_mat = multichannel_embedding_encoder(X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    train_mat = np.hstack((train_mat, demographics_train))#merge the demographics info

    lr = LinearRegression(normalize=True, n_jobs=-1)
    lr.fit(train_mat, y_train.ravel())
    test_mat = np.hstack((test_mat, demographics_test))#merge the demographics info
    y_pred = lr.predict(test_mat)
    r2, rmse = evaluation(y_test, y_pred)

    print >>f,'testing on filter testing samples...'
    rare_test_mat = multichannel_embedding_encoder(rare_X_test, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    rare_test_mat = np.hstack((rare_test_mat, rare_demographics_test))#merge the demographics info
    rare_y_pred = lr.predict(rare_test_mat)
    r2, rmse = evaluation(rare_y_test, rare_y_pred)

    print >>f,'testing on filter samples...'
    rare_mat = multichannel_embedding_encoder(rare_X, index_embedding, index_embedding1, index_embedding2, index_embedding3)
    rare_mat = np.hstack((rare_mat, rare_demographics))#merge the demographics info
    rare_y_pred = lr.predict(rare_mat)
    r2, rmse = evaluation(rare_y, rare_y_pred)


def train_single_channel_merge_model(model, modelpath, X_train, demographics_train, y_train):
    #train_mat,valid_mat, train_y, valid_y, demographics_train, demographics_valid = train_test_split(X_train, y_train, demographics_train, test_size=0.05, random_state=seed)
    checkpointer = ModelCheckpoint(filepath="./cnn_model/%s.hdf5"%(modelpath),verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    print 'Training the merge model....'
    model.fit([X_train, demographics_train], y_train, epochs=100, batch_size=args.batchsize, shuffle=True,
              validation_split=0.1,
              callbacks=[checkpointer,earlystopper],
              verbose=1)

def train_single_channel_split_model(model, modelpath, X1_train, X2_train, demographics_train, y_train):
    #train_mat,valid_mat, train_y, valid_y, demographics_train, demographics_valid = train_test_split(X_train, y_train, demographics_train, test_size=0.05, random_state=seed)
    checkpointer = ModelCheckpoint(filepath="./cnn_model/%s.hdf5"%(modelpath),verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    print 'Training the merge model....'
    model.fit([X1_train, X2_train, demographics_train], y_train, epochs=100, batch_size=args.batchsize, shuffle=True,
              validation_split=0.1,
              callbacks=[checkpointer,earlystopper],
              verbose=1)

def train_multi_channel_split_model(model, modelpath, X1_train, X2_train, demographics_train, y_train):
    #train_mat,valid_mat, train_y, valid_y, demographics_train, demographics_valid = train_test_split(X_train, y_train, demographics_train, test_size=0.05, random_state=seed)
    checkpointer = ModelCheckpoint(filepath="./cnn_model/%s.hdf5"%(modelpath),verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

    print 'Training the merge model....'
    model.fit([X1_train, X2_train, demographics_train], y_train, epochs=100, batch_size=args.batchsize, shuffle=True,
              validation_split=0.1,
              callbacks=[checkpointer,earlystopper],
              verbose=1)


def test_single_channel_merge_model(model, modelpath,X_test,demographics_test,y_test, index):
    print 'Testing model...'
    model.load_weights("./cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X_test, demographics_test], batch_size=args.batchsize, verbose=1)
    r2,rmse = evaluation(y_test, y_pred)
    name = 'SG_merge%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

def test_single_channel_split_model(model, modelpath, X1_test, X2_test, demographics_test, y_test,index):
    print 'Testing model...'
    model.load_weights("./cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X1_test, X2_test, demographics_test], batch_size=args.batchsize, verbose=1)
    r2,rmse = evaluation(y_test, y_pred)
    name = 'SG_merge%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)

def test_multi_channel_split_model(model, modelpath, X1_test, X2_test, demographics_test, y_test,index):
    print 'Testing model...'
    model.load_weights("./cnn_model/%s.hdf5"%(modelpath))
    y_pred = model.predict([X1_test, X2_test, demographics_test], batch_size=args.batchsize, verbose=1)
    r2,rmse = evaluation(y_test, y_pred)
    name = 'MG_split%d'%(index)
    if args.isdays:
        daysplots(y_test, y_pred, r2, rmse, name)
    else:
        costplots(y_test, y_pred, r2, rmse, name)


demgras_dim = demographics.shape[1]
print demgras_dim
#%s:%s_multichannel_model_bestmodel_%sisdays_%sinit_%strainable_%fdpt_%flr_%dfz_%dfn_%dmaxlen_%ddim_%sfilter_%stransform_%sbatch'%(MID,SID,args.isdays,args.init, args.trainable, args.dpt, args.lr, args.fz, args.fn, args.maxlen, args.dim, args.filter, args.transform,args.batchsize)
modelpath = 'SG_merge' + path
model = single_channel_merge_model(demgras_dim)
if not args.test:
    train_single_channel_merge_model(model, modelpath, X_train, demographics_train, y_train)
print >>f,'testing on the testing datasets.....'
test_single_channel_merge_model(model, modelpath, X_test, demographics_test, y_test, 3)
print >>f,'testing on the filtered testing datasets.....'
test_single_channel_merge_model(model, modelpath,rare_X_test, rare_demographics_test, rare_y_test,2)
print >>f,'testing on the filtered datasets.....'
test_single_channel_merge_model(model, modelpath,rare_X, rare_demographics, rare_y,1)

modelpath = 'SG_split' + path
model = single_channel_split_model(demgras_dim)
if not args.test:
    train_single_channel_split_model(model, modelpath, X1_train, X2_train, demographics_train, y_train)
print >>f,'testing on the testing datasets.....'
test_single_channel_split_model(model, modelpath, X1_test, X2_test, demographics_test,y_test,3)
print >>f,'testing on the filtered testing datasets.....'
test_single_channel_split_model(model, modelpath, rare_X1_test, rare_X2_test, rare_demographics_test, rare_y_test,2)
print >>f,'testing on the filtered datasets.....'
test_single_channel_split_model(model, modelpath, rare_X1, rare_X2, rare_demographics, rare_y,1)

modelpath = 'MG_merge' + path
model= multi_channel_split_model(demgras_dim)
#%s:%s_split_merge_model_bestmodel__%sisdays_%sinit_%strainable_%fdpt_%flr_%dfz_%dfn_%dmaxlen_%ddim_%sfilter_%stransform_%sbatch'%(MID, SID,args.isdays, args.init, args.trainable, args.dpt, args.lr, args.fz, args.fn, args.maxlen, args.dim, args.filter, args.transform, args.batchsize)
if not args.test:
    train_multi_channel_split_model(model, modelpath, X1_train, X2_train, demographics_train, y_train)
extract_patientvec(model, modelpath, disease, surgery, demographics)
print >>f,'testing on the testing datasets.....'
test_multi_channel_split_model(model, modelpath, X1_test, X2_test, demographics_test,y_test,3)
print >>f,'testing on the filtered testing datasets.....'
test_multi_channel_split_model(model, modelpath, rare_X1_test, rare_X2_test, rare_demographics_test, rare_y_test,2)
print >>f,'testing on the filtered datasets.....'
test_multi_channel_split_model(model, modelpath, rare_X1, rare_X2, rare_demographics, rare_y,1)
#"""

embedding_RF(X_train, X_test,y_train,y_test,demographics_train,demographics_test,rare_X_test,rare_demographics_test,rare_y_test,rare_X, rare_demographics, rare_y)
embedding_LR(X_train, X_test,y_train,y_test,demographics_train,demographics_test,rare_X_test,rare_demographics_test,rare_y_test,rare_X, rare_demographics, rare_y)
#onehot_RF(X_train, X_test,y_train,y_test,demographics_train,demographics_test,rare_X_test,rare_demographics_test,rare_y_test)
#sys.stdout = orig_stdout
f.close()
