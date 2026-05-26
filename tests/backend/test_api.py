import pytest
from fastapi import HTTPException
from types import SimpleNamespace


@pytest.fixture
def api_module(fake_model_factory):
    from backend import api as api_module

    return api_module


@pytest.fixture
def models(api_module, fake_model_factory):
    return api_module.load_all_required_models(factory=fake_model_factory)


@pytest.fixture
def fake_request(models):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(models=models)))


def test_health_returns_loaded_models(api_module, fake_request):
    body = api_module.health(fake_request)
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == {"short_text", "long_article", "general"}


def test_predict_short_text_returns_neutral(api_module, fake_request):
    body = api_module.predict(fake_request, api_module.TextRequest(text="too short"))
    assert body["label"] == "NEUTRAL"


def test_predict_uses_short_text_model_for_up_to_100_words(api_module, fake_request):
    text = "word " * 10
    body = api_module.predict(fake_request, api_module.TextRequest(text=text))
    assert body["label"] == "REAL"
    assert body["meta"]["used_dataset"] == "WELFake"
    assert body["meta"]["used_model"] == "xgb"


def test_predict_uses_long_article_model_for_over_100_words(api_module, fake_request):
    text = "word " * 150
    body = api_module.predict(fake_request, api_module.TextRequest(text=text))
    assert body["meta"]["used_dataset"] == "LIAR"


def test_predict_uses_short_text_model_for_100_words(api_module, fake_request):
    text = "word " * 100
    body = api_module.predict(fake_request, api_module.TextRequest(text=text))
    assert body["meta"]["used_dataset"] == "WELFake"


def test_predict_uses_short_text_model_for_previous_medium_text(api_module, fake_request):
    text = "word " * 60
    body = api_module.predict(fake_request, api_module.TextRequest(text=text))
    assert body["meta"]["used_dataset"] == "WELFake"


def test_text_request_rejects_oversized_text(api_module):
    with pytest.raises(ValueError):
        api_module.TextRequest(text="a" * 60_000)


def test_predict_503_when_no_models(api_module):
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(models={})))
    with pytest.raises(HTTPException) as exc:
        api_module.predict(request, api_module.TextRequest(text="word " * 50))
    assert exc.value.status_code == 503


def test_select_best_model_threshold_boundaries(api_module):
    short = type("M", (), {"name": "s"})()
    general = type("M", (), {"name": "g"})()
    long_ = type("M", (), {"name": "l"})()
    models = {"short_text": short, "general": general, "long_article": long_}

    assert api_module.select_best_model("a " * 10, models) is short
    assert api_module.select_best_model("a " * 100, models) is short
    assert api_module.select_best_model("a " * 101, models) is long_
    assert api_module.select_best_model("a " * 200, models) is long_
    assert api_module.select_best_model("anything", {}) is None
