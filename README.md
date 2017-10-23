# word2vec_Miyazawa
word2vec for Japanese language test. 
Train with several works of Kiichi Miyazawa (download from [Aozora Bunku](http://www.aozora.gr.jp/index_pages/person81.html#sakuhin_list_1).)<br />
But you can use any input you like. Support Japanese and other UTF-8 input.<br/>
(for some other non-ascii text, you need to import extra font file)

## Preprocess
'txr_preprocess.py' would output the segmented([Mecab](http://taku910.github.io/mecab/#download) required) and punctuation-free file.<br />
See 'miyazawa_raw.txt' and 'miyazawa_seg.txt' for the differece.

## Visualization
using T-SNE(sklearn) for dedimension and bokeh or plt for visualization.<br />
(you can use the pre-tained file 'miyazawa_vec.txt' to test.)

## Compare words
use `def top_similar(inp, voclist, vec)` and `word_analogy(w1, w2, w3, voclist, vec)` to compare word vectors.
