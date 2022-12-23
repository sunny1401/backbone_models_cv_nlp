from typing import Optional
import numpy as np
from src.nlp.preprocessing import DataPreprocessor
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer


class TopicModel:

    def __init__(
        self, 
        model_type: str, 
        data_processor: Optional[DataPreprocessor] = None,
        word_dict_save_path: Optional[str] = None
    ):

        if not data_processor:
            raise ValueError(
                "Need one of data_processor to proceed"
            )
        self._word_dict = None
        self._tokens = None
        self._embeddings = None
        self._word_dict_save_path = word_dict_save_path
        if model_type == "LDA" or model_type =="combined":
            data_processor.corpus["lemmatized_sentence"] = (
                data_processor.corpus.cleaned.apply(
                    lambda d: data_processor.lemmatize_text(d))
            )

            data_processor.corpus["tokens"] = (
                data_processor.corpus.lemmatized_sentence.apply(
                    lambda d: data_processor.generate_ngrams(d))
            )

            data_processor.corpus["tokens"] = (
                data_processor.corpus.tokens.apply(
                    lambda d: [" ".join(token) for token in d])
            )
        
            self._word_dict = Dictionary(data_processor.corpus.tokens.tolist())
            self._word_dict.filter_extremes(no_below=20, no_above=0.5)
            data_processor.corpus["embeddings"] = data_processor.corpus.tokens.apply(self._word_dict.doc2bow)
            self._tokens = data_processor.tokens.to_list()
            embeddings = data_processor.embeddings.to_list()
            

        elif model_type == "BERT":
            model = SentenceTransformer('bert-base-nli-max-tokens')
            embeddings = np.array(
                model.encode(data_processor.corpus.cleaned.tolist(), 
                show_progress_bar=True)
            )
        
        if model_type == "combined":
            pass

    def train(self):

        if self._model_type





