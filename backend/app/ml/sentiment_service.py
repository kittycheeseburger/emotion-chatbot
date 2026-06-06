import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = BACKEND_DIR / "models" / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from app.schemas.chat import EmotionResult


class SentimentService:
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else BACKEND_DIR / "models" / "sentiment_smp_roberta"
        self.model = None
        self.tokenizer = None
        self.device = None
        self.max_length = 128
        self.id_to_label = {0: "悲伤", 1: "快乐", 2: "爱", 3: "愤怒", 4: "恐惧", 5: "惊讶", 6: "平静"}
        self._load_attempted = False

    def analyze(self, text: str) -> EmotionResult:
        if self._ensure_model_loaded():
            return self._analyze_with_model(text)

        return self._analyze_with_rules(text)

    def _ensure_model_loaded(self) -> bool:
        if self._load_attempted:
            return self.model is not None and self.tokenizer is not None

        self._load_attempted = True
        model_path = self.model_dir / "model.pt"
        tokenizer_path = self.model_dir / "tokenizer"

        if not model_path.exists() or not tokenizer_path.exists():
            return False

        try:
            import torch
            from transformers import AutoTokenizer

            from app.ml.bert_sentiment import SentimentClassifier

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            config = checkpoint["model_config"]
            self.max_length = int(config.get("max_length", self.max_length))
            self.id_to_label = {int(key): value for key, value in config.get("id_to_label", self.id_to_label).items()}
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            self.model = SentimentClassifier(
                model_name=config.get("model_name", "bert-base-chinese"),
                num_labels=int(config.get("num_labels", 3)),
                local_files_only=True,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
        except Exception:
            self.model = None
            self.tokenizer = None

        return self.model is not None and self.tokenizer is not None

    def _analyze_with_model(self, text: str) -> EmotionResult:
        import torch

        from app.ml.bert_sentiment import preprocess_text

        encoding = self.tokenizer(
            preprocess_text(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs["logits"], dim=-1)[0]
            predicted_id = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[predicted_id].item())

        raw_label = self.id_to_label[predicted_id]
        keyword_adjusted = self._adjust_by_clear_keywords(text, raw_label, confidence)

        if keyword_adjusted:
            return keyword_adjusted

        label_map = {
            "快乐": ("积极", "😊", min(100, round(70 + confidence * 25))),
            "爱": ("积极", "😊", min(100, round(75 + confidence * 20))),
            "平静": ("平静", "🙂", round(50 + confidence * 20)),
            "悲伤": ("低落", "😔", max(0, round(45 - confidence * 25))),
            "愤怒": ("愤怒", "😠", max(0, round(42 - confidence * 25))),
            "恐惧": ("焦虑", "😟", max(0, round(45 - confidence * 25))),
            "惊讶": ("惊讶", "😮", round(55 + confidence * 20)),
        }
        label, emoji, score = label_map[raw_label]

        return EmotionResult(label=label, emoji=emoji, score=score, confidence=round(confidence, 4))

    def _adjust_by_clear_keywords(self, text: str, raw_label: str, confidence: float) -> EmotionResult | None:
        anxiety_keywords = ("焦虑", "担心", "害怕", "恐惧", "紧张", "不安", "慌")

        if any(keyword in text for keyword in anxiety_keywords):
            return EmotionResult(label="焦虑", emoji="😟", score=26, confidence=round(max(confidence, 0.84), 4))

        negative_event_keywords = (
            "考砸",
            "没考好",
            "考差",
            "考试失败",
            "挂科",
            "没过",
            "失败了",
            "搞砸",
            "弄砸",
            "失利",
        )

        if any(keyword in text for keyword in negative_event_keywords):
            return EmotionResult(label="低落", emoji="😔", score=28, confidence=round(max(confidence, 0.82), 4))

        calm_keywords = ("平静", "一般般", "还行", "正常", "没什么特别", "没啥特别")
        positive_markers = ("开心", "高兴", "喜欢", "幸运", "惊喜", "顺利", "很好")
        negative_markers = ("难过", "焦虑", "生气", "害怕", "痛苦", "崩溃", "烦")

        if (
            raw_label in {"快乐", "爱", "悲伤"}
            and any(keyword in text for keyword in calm_keywords)
            and not any(keyword in text for keyword in positive_markers + negative_markers)
        ):
            return EmotionResult(label="平静", emoji="🙂", score=65, confidence=round(max(confidence, 0.78), 4))

        return None

    def _analyze_with_rules(self, text: str) -> EmotionResult:
        normalized = text.lower()

        rules = [
            (("开心", "高兴", "顺利", "满意", "期待", "喜欢", "太好了", "不错", "完成"), "积极", "😊", 82, 0.72),
            (("焦虑", "担心", "压力", "紧张", "烦躁", "害怕", "不安", "卡住"), "焦虑", "😟", 42, 0.76),
            (("难过", "失落", "沮丧", "崩溃", "孤独", "委屈", "累", "疲惫"), "低落", "😔", 34, 0.74),
            (("生气", "愤怒", "火大", "讨厌", "烦死", "不公平"), "愤怒", "😠", 31, 0.73),
        ]

        for keywords, label, emoji, score, confidence in rules:
            if any(keyword in normalized for keyword in keywords):
                return EmotionResult(label=label, emoji=emoji, score=score, confidence=confidence)

        return EmotionResult(label="平静", emoji="🙂", score=68, confidence=0.58)
