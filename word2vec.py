import gensim

input_file='corpus/pd-aio.txt.gz'
output_file='word.w2v'



sentences = gensim.models.word2vec.LineSentence(input_file)
model = gensim.models.Word2Vec(sentences, size=300, min_count=10, sg=1, workers=4)
model.save(output_file)
