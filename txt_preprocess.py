#-*- encoding: utf-8 -*-
import MeCab
import sys
import codecs
import argparse
import re, string

parser =argparse.ArgumentParser()
parser.add_argument('input_dict', help = 'the file of input dictionary')
parser.add_argument('--utf8', help = 'switch to utf-8 mode', action = 'store_true')
args = parser.parse_args()

index = args.input_dict.find('.txt')
output_file = args.input_dict[:index] +'_seg'+ args.input_dict[index:]

input_file = open(args.input_dict, 'r')

#mecab for japanese segmentation
if args.utf8:
    lines = input_file.read().decode('utf-8').split(u'\n')
    lines = list(set(lines))
    #remove punctuation
    jpun = re.compile(u'^[\u3000-\u303f\uff00-\uff65\n\t\r]$')
    mt = MeCab.Tagger('-F\s%f[6] -U\s%m -E\\n')
    line_num = 0
    for line in lines:
        result = mt.parse(line.encode('utf-8'))
        for words in result.decode('utf-8').split('\n'):
            for char in words:
                if jpun.match(char) != None:
                    words = words.replace(' '+char, '')
            with codecs.open(output_file, 'a', 'utf-8') as output_txt:
                    output_txt.write(words+' ')
        with codecs.open(output_file, 'a', 'utf-8') as output_txt:
             output_txt.write('\n')
        line_num += 1
        if line_num % 100 ==0:
            print('line ', line_num, " processed")

else: #english text
    lines = input_file.read().split('\n')
    lines = list(set(lines))
    #remove punctuation
    pl = list(set(string.punctuation))
    pl.remove('-')
    pl.extend(['\n', '\r','\t'])
    print(pl)
    for line in lines:
        nline = []
        line_num = 0
        for ch in line:
            if ch not in pl:
                nline.append(ch)
        line = ''.join(ch for ch in nline)
        line = line.replace('--', ' ')
        with open(output_file, 'a') as output_txt:
             output_txt.write(line+'\n')

        line_num += 1
        if line_num % 100 ==0:
            print('line ', line_num, " processed")
