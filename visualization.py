from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from bokeh.plotting import figure
from bokeh.io import output_file,show
from bokeh.models import LabelSet,ColumnDataSource, HoverTool

def tsne(voclist, vec):
    
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress = True)
    return model.fit_transform(vec)


def tsne_plt(voclist, vec):
    fp =FontProperties(fname= './notofonts/NotoSansCJKjp-hinted/NotoSansCJKjp-DemiLight.otf', size = 12)
    v = tsne(voclist, vec)
    plt.scatter(v[:, 0], v[:,1])
    for label, x, y in zip(voclist, v[:, 0], v[:,1]):
        plt.annotate(label , xy= (x,y), xytext = (0,0), textcoords = 'offset points', fontproperties = fp)
    return plt.show()

def tsne_bokeh(voclist, vec):
    v = tsne(voclist, vec)
    toolbar = 'hover,save,pan,box_zoom,reset,wheel_zoom'
    source = ColumnDataSource(data=dict(x=v[:,0],y=v[:,1],la=voclist))
    p = figure(plot_height=1000, plot_width=2000, tools=toolbar)
    p.scatter(x='x', y='y',size=5, fill_alpha= 0.5, line_color=None, source=source)
    labels = LabelSet(x='x', y='y', text='la', x_offset=0, y_offset=0, level='glyph',source=source, )
    p.add_layout(labels)
    hover = p.select_one(HoverTool)
    hover.tooltips = [('Word','@la'),('x','$x'),('y','$y')]
    outpufile='w2c.html'
    show(p)

def main():
    pass

if __name__=='__main__':
    main()
