
from gensim.models.coherencemodel import CoherenceModel

def get_coherence_score(
    model_type, nlp_model, measure='u_mass'
):

    if model_type == "LDA":

        coherence_model = CoherenceModel(
            model=nlp_model.model,
            texts=nlp_model.tokens,
            corpus=nlp_model.embeddings,
            dictionary=nlp_model.word_dict,
            measure=measure
        )
    
        return coherence_model.get_coherence()


def get_silhoutte_score():
    pass