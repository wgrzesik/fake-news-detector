from research.configs.models.transformer_base import TransformerBaseModel


class BertModel(TransformerBaseModel):
    """BERT model for fake news detection (fine-tuning)."""

    DEFAULT_PRETRAINED = "bert-base-uncased"

    def __init__(self, dataset_name: str, embedding_type: str = "bert-base-uncased", **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            model_name=f"bert_{embedding_type}",
            embedding_type=embedding_type,
            **kwargs,
        )
