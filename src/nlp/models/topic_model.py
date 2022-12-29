from typing import Optional, Union, Dict
import numpy as np
from src.nlp.preprocessing import DataPreprocessor
from sklearn.cluster import KMeans
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer
from gensim.models.ldamodels import LdaModel
from gensim.models.ldamulticore import LdaMulticore


import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


class TopicModel:

    def __init__(
        self, 
        model_type: str, 
        data_processor: Optional[DataPreprocessor] = None,
        word_dict_save_path: Optional[str] = None,
        lda_bert_gamma: int = 20
    ):

        if not data_processor:
            raise ValueError(
                "Need one of data_processor to proceed"
            )
        self._word_dict = None
        self._tokens = None
        self._embeddings = dict()
        self._word_dict_save_path = word_dict_save_path
        self._lda_bert_gamma = lda_bert_gamma
        self._model = None
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
            self._embeddings = {
                "LDA": data_processor.embeddings.to_list()
            }

        elif model_type == "BERT" or model_type == "combined":
            model = SentenceTransformer('bert-base-nli-max-tokens')
            embeddings = np.array(
                model.encode(data_processor.corpus.cleaned.tolist(), 
                show_progress_bar=True)
            )
            self._embeddings = {"BERT": embeddings}

        self._model_type = model_type

    def train(
        self, 
        model_details: Dict, 
        beta_value_lda_model: Optional[Union[np.array, str]] = None,
        clustering_model: callable = KMeans,
        clustering_model_details: Optional[Dict] = None,
        distributed: bool = False,
        parallel: bool = False, 
    ):

        if self._model_type == "LDA" or self._model_type == "combined":

            data_details = dict(
                corpus=self._embeddings["LDA"], 
                id2word=self._word_dict.id2token
            )
            model_details = model_details | data_details
            
            if (parallel and beta_value_lda_model != "auto"):
                self._model = LdaMulticore(**model_details)
            else:
                model_details["distributed"] = distributed
                self._model = LdaModel(**model_details)

            self._embeddings["LDA_probability_vectors"] = self.get_LDA_probability_vectors(
                num_topics=model_details["num_topics"]
            )

        if self._model_type in {"BERT", "combined"}:
        
            if self._model_type == "combined":
                self._embeddings["LDA_BERT"] = np.c_[self._embeddings["LDA_probability_vectors"] * self._lda_bert_gamma, self._embeddings["BERT"]]
                default_embeddings = self._embeddings["LDA_BERT"]

            elif self._model_type == "BERT":
                default_embeddings = self._embeddings["BERT"]
                
            if clustering_model_details:
                clustering_model_details["n_clusters"] = model_details["num_topics"]
            else:
                clustering_model_details = dict(n_clusters=model_details["num_topics"])
            self._model = clustering_model(**clustering_model_details)
            self._model.fit(default_embeddings)


    def predict(self, test_data):
        pass

        

    def get_LDA_probability_vectors(self, num_topics):

        lda_probability_vector = np.zeros(
            (len(self._embeddings["LDA"]), num_topics))
        for i in range(len(self._embeddings)):
            # get the distribution for the i-th document in corpus
            for topic, prob in self._model.get_document_topics(
                self._embeddings["LDA"][i]
            ):
                lda_probability_vector[i, topic] = prob
            
        return lda_probability_vector
