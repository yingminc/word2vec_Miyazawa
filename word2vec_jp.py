#-*- encoding: utf-8 -*-
from __future__ import division
import math
import struct
import numpy as np
from multiprocessing import Pool, Value, Array

import scipy
import codecs



class ivoc:
    def __init__(self, word):
        self.word = word
        self.freq = 0

#process the vocabulary from the data
class Vocs:
    def __init__(self, input_file, min_fq):
        #load data
        input_file = open(input_file, 'r')
        lines = input_file.read().decode('utf-8').lower().split('\n')

        #list data and data info
        voc_list = [] #list of all voc
        voc_dict = {} #dict of voc index
        voc_rdict = {}
        line_list = []
        voc_count = 0
        word_count = 0
        for line in lines:
            word_list = []
            words = line.split(' ')
            for word in words:
                if word == (' ' or ''):
                    continue
                word_count += 1
                if word not in voc_dict:
                    voc_dict[word] = voc_count
                    voc_rdict[voc_count] = word
                    voc_count += 1
                    voc_list.append(ivoc(word))
                voc_list[voc_dict[word]].freq +=1 #assign the freqency to voc
                word_list.append(voc_dict[word])
            line_list.append(word_list)

        self.voc_list = voc_list
        self.dict = voc_dict
        self.rdict = voc_rdict
        self.word_count = word_count
        self.line_list = line_list
        self.fq_sort(min_fq)

    #discard the rare vocs and sort the list by freqency
    def fq_sort(self, min_fq):
        nvoc_list = []
        nvoc_list.append(ivoc('<unk>'))
        unk_list = []
        unk_index = 0
        unk_count = 0
        for i in self.voc_list:
            #if i is a rare voc
            if i.freq < min_fq:
                unk_count += 1
                nvoc_list[unk_index].freq += i.freq
                unk_list.append(self.dict[i.word])

            #if i is not rare voc: add it to new list
            else:
                nvoc_list.append(i)

        #sort the list according to freqency
        nvoc_list.sort(key=lambda voc: voc.freq, reverse=True)

        #renew the dict of voc index
        nvoc_rdict = {}
        nvoc_dict = {}
        for ind, i in enumerate(nvoc_list):
            nvoc_dict[i.word] = ind
            nvoc_rdict[ind] = i.word

        nline_list = []
        for line in self.line_list:
            word_list = []
            for iword in line:
                if iword in unk_list:
                    i = nvoc_dict['<unk>']
                    word_list.append(i)
                else:
                    word = self.rdict[iword]
                    i = nvoc_dict[word]
                    word_list.append(i)
            nline_list.append(word_list)

        self.line_list = nline_list
        self.voc_list = nvoc_list
        self.dict = nvoc_dict
        self.rdict = nvoc_rdict

    #call the index of the voc
    def index(self, voc):
        if voc in self.voc_list:
            return self.dict[voc]
        else:
            return self.dict['<unk>']

#unigramtable for negative sampling, generated for vocs
class unigramtable:
    def __init__(self,vocs):
        #rise the distribution by (3/4) power
        pw = 0.75

        #normalizing fraction
        nf = sum([math.pow(i.freq, pw) for i in vocs.voc_list])

        #create an emppty table
        table_size = int(1e7)
        table = np.zeros(table_size, dtype=np.uint32)

        #fill the table with index
        p = 0 # Cumulative probability
        i = 0
        for voc in vocs.voc_list:
            p += float(math.pow(voc.freq, pw))/nf
            while i < table_size and float(i) / table_size < p:
                table[i] = vocs.dict[voc.word]
                i += 1
        self.table = table

    #pick the vocs for negative sampling randomly
    def neg_sample(self, neg_num):
        indices = np.random.randint(0, len(self.table), size = neg_num)

        return [self.table[i] for i in indices] #return the indices of target vocs

#set initial weight for hidden layer
def init_w(dim, voc_size): #dimentions and number of voc
    random_0 = np.random.uniform(-0.5/dim, 0.5/dim, (voc_size, dim))

    #use ctypeslib to speed up
    syn0 = np.asarray(random_0)


    random_1 = np.zeros((voc_size, dim))
    syn1 = np.asarray(random_1)

    return (syn0, syn1)

def sigm(z):
    if z > 6:
        return 1.0
    elif z < -1:
        return 0.0
    else:
        return 1/(1 + math.exp(-z))

def train(input_file):
    #formalize the input_file
    min_freq = 3
    neg_num = 20
    dim = 100
    vocs = Vocs(input_file, min_freq)
    voc_size = len(vocs.voc_list)

    #set initial net
    syn0, syn1 = init_w(dim, voc_size)
    table = unigramtable(vocs)

    alpha = 0.05
    window_size = 10

    word_processed=0
    print 'making data'
    for line_num, line in enumerate(vocs.line_list):
        for pos, word in enumerate(line):
            #make dataset(x,y) with skipgram with randam window size for each x
            current_window = np.random.randint(1, window_size)
            dataset = []

            for i in range(1,current_window):
                if pos-i >= 0:
                    dataset.append((word, line[pos-i]))
                if pos+i <= len(line)-1:
                    dataset.append((word, line[pos+i]))

            for x,y in dataset:
                #negative sampling
                classifiers = [(y,1)]+[(neg,0) for neg in table.neg_sample(neg_num)]

                neule = np.zeros(dim)

                for y , tag in classifiers:
                    z = np.dot(syn0[x],syn1[y])
                    p = sigm(z)
                    g = alpha*(tag-p) #loss
                    neule += g*syn1[y] #save loss for backpropagate
                    syn1[y]+= g*syn0[x] #update syn1

                syn0[x] +=neule

            word_processed+=1

    index = input_file.find('.txt')
    output_file = input_file[:index] +'_vec'+ input_file[index:]

    with codecs.open(output_file, 'w', 'utf-8') as op:
        op.write('%d %d\n' % (len(syn0), dim))
        for ivoc, vector in zip(vocs.voc_list, syn0):
            word = ivoc.word
            vector_str = ' '.join([str(v) for v in vector])
            op.write('%s %s\n' % (word, vector_str))



    print 'done'

    return vocs, syn0


def pair_similar(w1, w2, voclist, vec):
    nw1 = voclist.index(w1)
    nw2 = voclist.index(w2)
    sim = 1 - scipy.spatial.distance.cosine(vec[nw1], vec[nw2])
    return sim


def top_similar(inp, voclist, vec, num=20):
    inpn = voclist.index(inp)
    sims = [1-scipy.spatial.distance.cosine(vec[inpn], i) for i in vec]
    wsim = zip(voclist, sims)
    swsim = sorted(wsim, key=lambda w: w[1], reverse = True)
    return swsim[:num]

def word_analogy(w1, w2, w3, voclist, vec):
    #w1-w2+w3
    xlist=[]
    for x in top_similar(w1, voclist, vec, num= (len(voclist)-1)):
        if pair_similar(x[0],w3,voclist,vec) > pair_similar(x[0],w2,voclist,vec):
            xlist.append(x)
    print xlist[:20]

def make_voclist(vocs):
    voclist = [ i.word for i in vocs.voc_list]
    return voclist

def load_vec(file):
    input_file= open(file, 'r')
    lines = input_file.read().split('\n')
    voc_num, dim = map(int, lines[0].split(' '))
    lines = lines[1:]
    voclist = [line.split(' ')[0].decode('utf-8') for line in lines]
    veclist = []
    for line in lines:
        veclist.extend(line.split(' ')[1:])
    vec = np.array(veclist)
    vec = np.reshape(vec,(voc_num, dim))
    return voclist, vec

def main():
    pass

if __name__=='__main__':
    main()
