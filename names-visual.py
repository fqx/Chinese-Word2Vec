import gensim,matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
#matplotlib.rcParams['font.family'] = 'sans-serif'


model_loc = 'word.w2v'
names_loc = 'names.txt'

model = gensim.models.Word2Vec.load(model_loc)

names_file = open(names_loc, 'r', encoding='utf-8')
names = names_file.readlines()
names_file.close()

tags = []
vecs = []

for name in names:
    name = name.strip()
    try:
        name_vec = model.wv[name]
        tags.append(name)
        vecs.append(name_vec)
    except:
        print('%s is not found' % name)


def plot_with_labels(low_dim_embs, labels, filename='tsne.svgz'):
    assert low_dim_embs.shape[0] <= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(36, 36))
    for i in range(len(labels)):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(labels[i], xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(filename)


tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(vecs)
plot_with_labels(low_dim_embs, tags)