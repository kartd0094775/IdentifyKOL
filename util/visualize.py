import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba
import subprocess 
from gensim.test.utils import get_tmpfile, common_texts
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontManager 
from pylab import mpl

def display_closestwords_tsnescatterplot(model, words):
    mpl.rcParams['font.sans-serif'] = 'DFKai-SB'
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    arr = np.empty((0,100), dtype='f')
    word_labels = list(map(lambda word: word, words))
    # color_labels= list()
    cnt = 0
    for word in words:
        # get close words
        close_words = model.similar_by_word(word)
        print(close_words)
        # add the vector for each of the closest words to the array
        arr = np.append(arr, np.array([model[word]]), axis=0)
        for wrd_score in close_words:
            wrd_vector = model[wrd_score[0]]
            # color_labels.append(colors[cnt])
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)
        cnt += 1
    print(word_labels)
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, n_iter=1000, random_state=0)
    #np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
        # display scatter plot
    plt.scatter(x_coords, y_coords)
    
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
    return word_labels, Y
def get_matplot_zh_font(): 
    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist) 
    output = subprocess.check_output('fc-list :lang=zh -f "%{family}\n"', shell=True)
    zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n')) 
    available = list(mat_fonts & zh_fonts)
    return available
def set_matplot_zh_font():
    available = get_matplot_zh_font() 
    if len(available) > 0:
        mpl.rcParams['font.sans-serif'] = [available[0]] # 指定默认字体
        mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
