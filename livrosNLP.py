import gensim
import nltk
import glob
import re
import codecs
import math
import scipy
import numpy


def createModels(filenames):
	#filenames = ['books/sherlock_adventures.txt', 'books/sherlock_baskervilles.txt']
	#filenames = ['books/sherlock_adventures.txt', 'books/plato_republic.txt']
	#filenames = ['books/sherlock_adventures.txt', 'books/sherlock_adventures.txt']
	tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

	models = []


	for fname in filenames:
		print("Current file: '{0}'".format(fname))
		
		with codecs.open(fname, "r", "utf-8") as fin:
			corpus = fin.read()
		#fin = open(fname, 'r')
		#corpus = fin.read()
		
		sentences = tokenizer.tokenize(corpus)
		#print("Sentences: '{0}'".format(sentences))

		words = []
		for sentence in sentences:
			cleanSentence = re.sub("[^a-zA-Z]", " ", sentence)		
			words.append(cleanSentence.lower().split())

		#print("Words: '{0}'".format(words))
		count = len(words)
		#print("count = '{0}'".format(count))

		model = gensim.models.Word2Vec(sg=1, seed=123, workers=1, size=100, window=10, min_count=10)
		
		model.build_vocab(words)
		model.train(words, total_examples=model.corpus_count, epochs=model.iter)
		models.append(model)

	return models


''' #first candidate for distance
distance = 0
count = 0
for model1 in models:
	for model2 in models:
		for word1 in model1.wv.vocab:
			for word2 in model2.wv.vocab:
				if word1==word2:
					#print("m1 w1 '{0}'".format(model1.wv[word1]))
					#print("m2 w2 '{0}'".format(model2.wv[word2]))
					distance += 1-scipy.spatial.distance.cosine(model1.wv[word1], model2.wv[word2])
					count += 1
					#print("distance '{0}'".format(distance))
distance = distance/count
print("final distance '{0}'".format(distance))
'''

def findDistance(model1, model2):
	distanceMat = []
	count = 0
	distance = 0
	for word1 in model1.wv.vocab:
		for word2 in model2.wv.vocab:
			if word1==word2:
				#print("m1 w1 '{0}'".format(model1.wv[word1]))
				#print("m2 w2 '{0}'".format(model2.wv[word2]))
				w1 = numpy.linalg.norm(model1.wv[word1])
				w2 = numpy.linalg.norm(model2.wv[word2])
				#distance += numpy.power((w1-w2),2)
				distance += abs(w1-w2)
				count += 1
				#print("distance '{0}'".format(distance))
				#distanceMat[models.index(model1)][models.index(model2)] = math.sqrt(distance)
	return 100*distance/count

def main():
	distMat = []
	filenames = glob.glob("books/*.txt")
	models = createModels(filenames)
	count = 0
	for fname1 in filenames:
		distRow = []
		for fname2 in filenames:
			id1 = models[filenames.index(fname1)]
			id2 = models[filenames.index(fname2)]
			#if id2>=id1:
			#	continue
			dist = findDistance(id1, id2)
			distRow.append(dist)
			print("Distance between \"" + fname1 + "\" and \"" + fname2 +"\" : "+ str(dist))
		distMat.append(distRow)
	print(distMat)

if __name__ == "__main__":
	main()