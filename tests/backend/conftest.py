import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_model_factory():
    """Stub out research.* before importing backend.api."""
    research = types.ModuleType("research")
    configs = types.ModuleType("research.configs")
    models_pkg = types.ModuleType("research.configs.models")

    base_module = types.ModuleType("research.configs.models.base_model")
    class BaseModel:  # minimal duck-typed stand-in
        dataset_name = "stub"
        model_name = "stub"
        def load(self): pass
        def predict(self, text): return {"label": "REAL", "score": 0.99}
    base_module.BaseModel = BaseModel

    factory_module = types.ModuleType("research.configs.models.model_factory")

    class FakeModel(BaseModel):
        def __init__(self, dataset, model_type, emb):
            self.dataset_name = dataset
            self.model_name = model_type
            self.emb = emb
            self.predict = MagicMock(return_value={"label": "REAL", "score": 0.88})

    class ModelFactory:
        @staticmethod
        def get_model(dataset, model_type, emb):
            return FakeModel(dataset, model_type, emb)

    factory_module.ModelFactory = ModelFactory

    sys.modules["research"] = research
    sys.modules["research.configs"] = configs
    sys.modules["research.configs.models"] = models_pkg
    sys.modules["research.configs.models.base_model"] = base_module
    sys.modules["research.configs.models.model_factory"] = factory_module

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    yield ModelFactory

    for mod in [
        "backend.api",
        "research.configs.models.model_factory",
        "research.configs.models.base_model",
        "research.configs.models",
        "research.configs",
        "research",
    ]:
        sys.modules.pop(mod, None)