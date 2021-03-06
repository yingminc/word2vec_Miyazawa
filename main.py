from visualization import *
from word2vec_jp import train, make_voclist, load_vec
import argparse

parser =argparse.ArgumentParser()
parser.add_argument('input', help = 'the file of input text')
parser.add_argument('--w2v', help = 'train vectors; else the input would be pretained vectors', action = 'store_true')
parser.add_argument('--vis_plt', help = 'visualization with plt', action = 'store_true')
parser.add_argument('--vis_bokeh', help = 'visualization with bokeh', action = 'store_true')
args = parser.parse_args()

if args.w2v:
    vocs, vec = train(args.input)
    voclist = make_voclist(vocs)
else:
    voclist, vec = load_vec(args.input)

if args.vis_plt:
    tsne_plt(voclist, vec)

if args.vis_bokeh:
    tsne_bokeh(voclist, vec)
