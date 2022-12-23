
from gensim.models.coherencemodel import CoherenceModel

def get_coherence_score(
    model_type, nlp_model, measure='c_v'
):

    if model_type == "SALDA":

        coherence_model = CoherenceModel(
            model=nlp_model.model,
            texts=nlp_model.tokens,
            corpus=nlp_model.embeddings,
            dictionary=nlp_model.word_dict,
            measure='c_v'
        )
    
        return coherence_model.get_coherence()
