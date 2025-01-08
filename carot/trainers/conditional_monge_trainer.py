from typing import Dict

from cmonge.models.embedding import BaseEmbedding
from cmonge.trainers.conditional_monge_trainer import (
    ConditionalMongeTrainer as _ConditionalMongeTrainer,
)

from carot.models.embedding import EmbeddingFactory


class ConditionalMongeTrainer(_ConditionalMongeTrainer):
    embedding_factory: Dict[str, BaseEmbedding] = EmbeddingFactory
