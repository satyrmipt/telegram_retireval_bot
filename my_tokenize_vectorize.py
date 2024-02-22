'''
This modula was created as my first python module during winter 2024 as part
of "NLP generative models" course homework. This is also first try to create
my own classes (and make them useful).


class Tokenizer implements several methods of primitive tokenization.
class Vectorizer implements methods of primitive vectorizatio.

Both classes are just wrappers for ready-to-use ,ethods of well-known
python libraries.

Первые шаги такие первые шаги.

'''

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize

class Tokenizer:

  def __init__(self, tokenizer: str = 'split', lower: bool = True, lang: str = 'english'):
    '''
    class initializaton.
    inputs:
      corpus: list of stings to tokenize
      tokenizer: one of 
           - 'split' (str.split())
           - 'word_tokenize' (nltk)
           - 'wordpunct_tokenize' (nltk)
           - 'sent_tokenize' (nltk)
           - 'bm25': custom tokenizer for bm25 algorithm
           - 'dummy' (do nothing with text, it was introduced to make code more universal)
      lower_ind: wheather to lowercase the strings before tokenization
      lang: some tokenizers has language options in parameters, use value for your particular tokenizer
    '''
    self.corpus=[]
    self.tokenizer=tokenizer
    self.tokenized_corpus = []
    self.lower = lower
    self.lang = lang
 
  def tokenize_corpus(self, corpus: list[str] ) -> list[list[str]] | list[str]:
    '''
    lowercase (optionally) and tokenize list of strings to list of list of words or ngrmas or tokens
    inputs:
      corpus: list of stings to tokenize
    '''
    if  self.lower:
      self.corpus = [str(sent).lower() for sent in corpus] #str protects from np.nan
    else:
      self.corpus = corpus
    # dummy tokenizer for model wich do not need tokenization
    if self.tokenizer == 'dummy':
        return corpus
    elif self.tokenizer == 'split':
      return [sent.split() for sent in self.corpus]
    elif self.tokenizer == 'bm25':
      import string
      return [[w.strip(string.punctuation) for w in sent.lower().split()] for sent in self.corpus]
    elif self.tokenizer == 'word_tokenize':
      return [word_tokenize(sent, language=self.lang) for sent in self.corpus]
    elif self.tokenizer == 'wordpunct_tokenize':
      if self.lang != 'english':
        raise ValueError(f'{self.tokenizer} implemented for English language only')
      else:
        return [wordpunct_tokenize(sent) for sent in self.corpus]
    elif self.tokenizer == 'sent_tokenize':
      return [sent_tokenize(sent, language=self.lang) for sent in self.corpus]
    else:
      raise ValueError(f"{self.tokenizer=} is incorrect tokenizer name, see doc for full name list")

# corp = [
#     'But she, she heard the violin...',
#     'And left my side. And entered in',
#     'love paased into the house of lust!'
# ]
# tk = Tokenizer(tokenizer = 'split', lower=True, lang = 'english' )
# tk.tokenize_corpus(corp)
# print(tk.corpus, tk.tokenizer, tk.tokenized_corpus, tk.lower, tk.lang )

import pandas as pd
import numpy as np
import csv
import os

class Glove_vectorizer():

    def __init__(self, dim: int, path_to_emb_folder: str):
        self.col_prefix = 'glove_' # prefix for columns in dataset
        self.dim = dim
        self.path = os.path.join(path_to_emb_folder, f"glove.6B.{self.dim}d.txt")
        edf = pd.read_csv(self.path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
        edf_array = edf.to_numpy()
        self.emb = {k:v for (k, v) in zip(edf.index, edf_array )}   
        self.zero_vect = np.zeros(self.dim)
        del edf_array 
        del edf
        
    def fine_tune_glove(self):
        print('Not implemented yet') # see https://gist.github.com/chmodsss/867e01cc3eeeaa42226ac931709077dc

        
    def vectorize_tokens(self, tokenized_corpus: list[list[str]], aggregation_type :str = 'avg') -> list:
        '''
        transform list of list of words to list of single glove vector of dim size
        inputs:
            - text: list of words (tokenized string)
            - dim: size of every word vector in stanford glove pack http://nlp.stanford.edu/data/glove.6B.zip, one of 50, 100, 200, 300
            - aggregation_type: how to aggregate word vectors to single vector, only avg is implemented (element-wise aggregation)
        '''
        # zero dummy vectro for OOV words
        # (firmly saying we have to calculate average vector over glove dict)
        vectror_corp_word = [[self.emb.get(word.lower(), self.zero_vect) for word in sent] for sent in tokenized_corpus]
        if aggregation_type == 'avg':
            # exclude oov during averaging since they will bias average to zero
            vectror_corp_sent = [np.mean(vect_sent, axis=0) for vect_sent in vectror_corp_word if not (vect_sent==self.zero_vect).all()]
        else:
            raise ValueError(f"'{aggregation_type}' is not implemented yet, use 'avg' instead")
        if len(vectror_corp_sent) == 0:
            return [self.zero_vect]
        else:
            return vectror_corp_sent
    
    def clear_class_memory(self):
        self.dim = -99
        self.path = ''
        self.emb = dict()     


# v = Glove_vectorizer(dim = 50, path_to_emb_folder = "C:\\Users\\satyr\\Documents\\edu\\nlp2\\hw1\\data")
# v.vectorizer([['we','are', 'the','champions'],['codito', 'ergo', 'sum']])


from sentence_transformers import SentenceTransformer

class SentBERT_vectorizer():
    '''
    This vectorizer transforms sentences to vectros of given size wich is sutable
    for semantic comparison like cosine similarity.
    
    for more information see:
        https://github.com/UKPLab/sentence-transformers/blob/master/README.md
        https://www.sbert.net/docs/pretrained_models.html
        https://stackoverflow.com/questions/60492839/how-to-compare-sentence-similarities-using-embeddings-from-bert
        https://arxiv.org/abs/1908.10084
    '''
    def __init__(self, checkpoint: str = 'all-mpnet-base-v2'):
        self.col_prefix = 'sbert_' # prefix for columns in dataset
        self.checkpoint = checkpoint # see https://www.sbert.net/docs/pretrained_models.html for list of pretrainde models
        self.model = SentenceTransformer(model_name_or_path = checkpoint)
        self.dim = self.model.encode(['hello world'])[0].shape[0]
        self.zero_vect = np.zeros(self.dim)
        
    def vectorize_tokens(self, corpus: list[str]):
        '''
        IN FACT IT TOKENIZE CORPUST, NOT TOKENS!
        input: list of strings (sentences)
        output: 2D numpy array, one vectro for each sentece
        '''
        return self.model.encode(corpus)
    
    def fine_tune_sbert_triple(self):
        print('Not implemented yet')
    
    def clear_class_memory(self):
        del self.model  
        
from sentence_transformers import CrossEncoder

class Reranker():
    '''
    Rerank previously ranked results
    https://www.sbert.net/examples/applications/retrieve_rerank/README.html
    '''
    def __init__(self, checkpoint: str = 'model\\crerankingeval-30e-4000-ms-marco-MiniLM-L-6-v2'):
        self.checkpoint = checkpoint # see https://www.sbert.net/docs/pretrained-models/ce-msmarco.html for list of pretrainde models
        self.model = CrossEncoder(model_name = checkpoint)

    def rerank_results(self, results: list[list[str]]):
        return self.model.predict(sentences = results)