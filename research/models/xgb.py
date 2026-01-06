from xgboost import XGBClassifier
from research.base import BaseFakeNewsModel
from research.embeddings.base_embedding import BaseEmbedder

class XGBoostModel(BaseFakeNewsModel):
    def __init__(self, dataset_name: str, embedding_type: str):
        super().__init__(dataset_name, f"xgb_{embedding_type}")
        
        self.embedder = BaseEmbedder.create(embedding_type)
        self.classifier = XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        self.scaler = None