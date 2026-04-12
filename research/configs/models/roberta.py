from research.configs.models.transformer_base import TransformerBaseModel


class RobertaModel(TransformerBaseModel):
    """RoBERTa model for fake news detection (fine-tuning)."""

    DEFAULT_PRETRAINED = "roberta-base"

    def __init__(self, dataset_name: str, embedding_type: str = "roberta-base", **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            model_name=f"roberta_{embedding_type}",
            embedding_type=embedding_type,
            **kwargs,
        )
