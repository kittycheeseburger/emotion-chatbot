from app.ml.sentiment_service import SentimentService


def test_clear_academic_setback_is_low_mood():
    result = SentimentService().analyze("我今天考砸了")

    assert result.label == "低落"
    assert result.score < 40


def test_clear_neutral_statement_is_calm():
    result = SentimentService().analyze("没什么特别的，今天还算平静")

    assert result.label == "平静"
    assert 50 <= result.score <= 75
