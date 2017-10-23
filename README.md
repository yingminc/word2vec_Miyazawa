# word2vec_Miyazawa
word2vec for Japanese language test. 
Train with several works of Kiichi Miyazawa (download from Aozora Bunku).
But you can use any input you like. Support Japanese and other UTF-8 input(for some other non-ascii text, you need to import extra font file).

## Preprocess
'txr_preprocess.py' would output the segmented and punctuation-free file.
See 'miyazawa_raw.txt' and 'miyazawa_seg.txt' for the differece.

## Visualization
using T-SNE(sklearn) for dedimension and bokeh or plt for visualization.
(you can use the pre-tained file 'miyazawa_vec.txt' for test.

## Compare words
use `def top_similar(inp, voclist, vec)` and `word_analogy(w1, w2, w3, voclist, vec)` to compare word vectors.
