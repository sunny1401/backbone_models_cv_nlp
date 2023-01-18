from dataclasses import dataclass
from typing import Optional, List, Callable


@dataclass
class TopicModelLDAGeneral:

    alpha: float
    minimum_probability: float
    num_topics: int
    passes: int
    per_word_topics: bool
    random_state: int
    beta: int = None
    callbacks: Optional[List[callable]] = None
    chunksize: int = 2000
    decay: float = 0.5
    distributed: bool = False
    eval_every: int = 10
    interations: int = 50
    gamma_threhold: float = 0.001
    offset: float = 1.0
    minimum_phi_value: float = 0.01
    workers: int = 4
    update_every: int = 1
    