from research.configs.models.transformer_base import TransformerBaseModel
class DistilBertModel(TransformerBaseModel):
    """DistilBERT model for fake news detection (fine-tuning distilbert-base-uncased)."""
    DEFAULT_PRETRAINED = "distilbert-base-uncased"
    def __init__(self, dataset_name: str, embedding_type: str = "distilbert-base-uncased", **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            model_name=f"distilbert_{embedding_type}",
            embedding_type=embedding_type,
            **kwargs,
        )
