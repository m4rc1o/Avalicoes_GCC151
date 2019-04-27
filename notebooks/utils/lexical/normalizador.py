import nltk
from nltk.tokenize import word_tokenize
import unidecode
import string

class Normalizador:

	def __init__(self):
		self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
		self.word_tokenizer = word_tokenize
		self.stemmer = nltk.stem.RSLPStemmer()
		self.stopwords = nltk.corpus.stopwords.words('portuguese')


	def remove_punctuation(self, text):
	    return text.translate(str.maketrans('', '', string.punctuation))


	def remove_accents(self, text):
		return unidecode.unidecode(text)


	def to_lowercase(self, text):
		return text.lower()


	def tokenize_sentences(self, text):
	    return self.sent_tokenizer.tokenize(text)


	def tokenize_words(self, text):
	    return self.word_tokenizer(text)


	def remove_stopwords(self, tokens):
		return [token for token in tokens if token not in self.stopwords]


	def stemmize(self, tokens):
		stemized_words = []

		for word in tokens:
			stemized_words.append(self.stemmer.stem(word))

		return stemized_words
