from visualization import *
from word2vec_test_jp import train, make_voclist
import argparse

parser =argparse.ArgumentParser()
parser.add_argument('input', help = 'the file of input text')
parser.add_argument('--w2v', help = 'train vectors; else the input would be pretained vectors', action = 'store_true')
parser.add_argument('--vis_plt', help = 'visualization with plt', action = 'store_true')
parser.add_argument('--vis_bokeh', help = 'visualization with bokeh', action = 'store_true')
args = parser.parse_args()

if w2v:
    vocs, vec = train(args.input)
    voclist = make_voclist(vocs)
else:
    voclist, vec = load_vec(args.input)

if vis_plt:
    tsne_plt(voclist, vec)

if vis_bokeh:
    tsne(voclist, vec)
