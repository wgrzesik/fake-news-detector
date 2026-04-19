from research.configs.models.transformer_base import TransformerBaseModel


class FakeBertModel(TransformerBaseModel):
    """FakeBERT model for fake news detection (fine-tuning bert-base-uncased).

    Based on the FakeBERT architecture proposed for fake news classification.
    Uses ``bert-base-uncased`` as the pretrained backbone.
    """

    DEFAULT_PRETRAINED = "bert-base-uncased"

    def __init__(self, dataset_name: str, embedding_type: str = "bert-base-uncased", **kwargs):
        super().__init__(
            dataset_name=dataset_name,
            model_name=f"fakebert_{embedding_type}",
            embedding_type=embedding_type,
            **kwargs,
        )

