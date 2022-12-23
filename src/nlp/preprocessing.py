from typing import List

from gensim.utils import simple_preprocess
from nltk.util import everygrams, ngrams
from nltk.tokenize import RegexpTokenizer
from src.nlp.stopwords import english_stopwords
import warnings
warnings.simplefilter("ignore", UserWarning)
import spacy
import pandas as pd

spacy.prefer_gpu()

class DataPreprocessor:

    nlp = spacy.load("en_core_web_trf")
    
    def __init__(
        self, corpus: List[str], 
        n_grams: int = 3, 
        generate_multigrams: bool = True):

        self._ngrams = n_grams
        self._generate_multigrams = generate_multigrams
        self.corpus = pd.DataFrame(corpus, columns=["raw_text"])
        self.corpus["cleaned"] = self.corpus.raw_text.apply(
            lambda d: self._clean_input_data(d))


    def lemmatize_text(
        self, 
        text, 
        allowed_tags: List = ['NOUN', 'ADJ', 'VERB', 'ADV'],
        use_pos_tags: bool = True
    ) -> str:
        tokens = simple_preprocess(text, deacc=True)
        tokens = self.nlp(" ".join(tokens))
        if use_pos_tags:
            lemmatized_sentence = " ".join(
                [
                    token.text
                    for token in tokens 
                    if token not in english_stopwords and  token.pos_ in allowed_tags
                ])
        else:
            lemmatized_sentence = " ".join(
                [
                    token.text
                    for token in tokens 
                    if token not in english_stopwords 
                ])
        return lemmatized_sentence

    def _clean_input_data(self, text):
        text = text.replace('\n',' ')
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        return " ".join(text)

    def generate_ngrams(self, sentence) -> List:

        if self._ngrams > 5:
            raise ValueError("Currently only a max of 5-grams are supported")
        elif self._ngrams == 5:
            tokens  = list(everygrams(sentence.split(" ")))
        else:
            tokens = []

            if self._generate_multigrams:
                for i in range(1, self._ngrams+1):
                    tokens += ngrams(sentence.split(" "), i)
            else:
                tokens = ngrams(sentence.split(" "), self._ngrams)

        return tokens
