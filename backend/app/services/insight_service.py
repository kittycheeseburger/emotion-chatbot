import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.chat import ChatMessage, EmotionInsight, EmotionResult

BACKEND_DIR = Path(__file__).resolve().parents[2]
STRATEGY_PATH = BACKEND_DIR / "app" / "data" / "emotion_strategies.json"


class InsightService:
    def __init__(self, strategy_path: Path = STRATEGY_PATH) -> None:
        self.strategy_path = strategy_path
        self.strategies = self._load_strategies()
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        self.strategy_matrix = self.vectorizer.fit_transform(
            [self._strategy_document(strategy) for strategy in self.strategies]
        )

    def build(
        self,
        message: str,
        emotion: EmotionResult,
        history: list[ChatMessage] | None = None,
        review_mode: bool = False,
    ) -> EmotionInsight:
        history = history or []
        trigger = self._detect_trigger(message)
        strategy, retrieval_score = self._retrieve_strategy(message, emotion, trigger, history)
        summary = self._build_summary(message, emotion, trigger, history, strategy, review_mode)

        return EmotionInsight(
            trigger=trigger,
            goal=strategy["goal"],
            next_action=strategy["next_action"],
            tone=strategy["tone"],
            summary=summary,
            strategy=strategy["title"],
            retrieval_score=round(retrieval_score, 4),
            source="rag",
        )

    def build_reply(self, message: str, emotion: EmotionResult, insight: EmotionInsight) -> str:
        opener = self._reply_opener(emotion)
        context_sentence = (
            f"我理解你现在主要被“{insight.trigger}”影响，"
            f"当前情绪更接近{emotion.emoji} {emotion.label}。"
        )
        strategy_sentence = (
            f"这时可以先按“{insight.strategy}”来处理：{insight.next_action}"
        )
        follow_up = self._reply_follow_up(emotion, insight)
        return f"{opener}{context_sentence}{strategy_sentence}{follow_up}"

    def _load_strategies(self) -> list[dict[str, object]]:
        with self.strategy_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _strategy_document(self, strategy: dict[str, object]) -> str:
        parts = [
            str(strategy["title"]),
            " ".join(strategy["emotions"]),
            " ".join(strategy["triggers"]),
            " ".join(strategy["keywords"]),
            str(strategy["goal"]),
            str(strategy["tone"]),
            str(strategy["next_action"]),
            str(strategy["summary_template"]),
        ]
        return " ".join(parts)

    def _retrieve_strategy(
        self,
        message: str,
        emotion: EmotionResult,
        trigger: str,
        history: list[ChatMessage],
    ) -> tuple[dict[str, object], float]:
        recent_context = " ".join(item.content for item in history[-4:] if item.role == "user")
        query = f"{message} {recent_context} {emotion.label} {trigger} 情绪指数{emotion.score}"
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.strategy_matrix)[0]

        boosted_scores = []
        for index, strategy in enumerate(self.strategies):
            score = float(scores[index])
            if emotion.label in strategy["emotions"]:
                score += 0.18
            if trigger in strategy["triggers"]:
                score += 0.12
            if any(keyword in message for keyword in strategy["keywords"]):
                score += 0.12
            boosted_scores.append(score)

        best_index = max(range(len(boosted_scores)), key=boosted_scores.__getitem__)
        return self.strategies[best_index], min(boosted_scores[best_index], 1.0)

    def _detect_trigger(self, text: str) -> str:
        trigger_rules = [
            (("考砸", "没考好", "考试", "挂科", "成绩", "复习"), "学业受挫"),
            (("工作", "项目", "老板", "同事", "加班", "汇报", "任务"), "工作压力"),
            (("吵架", "朋友", "家人", "恋人", "被骂", "关系", "沟通"), "人际关系"),
            (("睡不着", "失眠", "累", "疲惫", "身体", "头疼"), "身心疲惫"),
            (("未来", "不知道", "迷茫", "选择", "决定", "担心"), "不确定感"),
        ]

        for keywords, trigger in trigger_rules:
            if any(keyword in text for keyword in keywords):
                return trigger

        return "日常情绪表达"

    def _reply_opener(self, emotion: EmotionResult) -> str:
        openers = {
            "低落": "听起来这件事确实让你很受挫。先不用急着否定自己，",
            "焦虑": "这种卡住和不确定感会让人很紧绷。我们先把问题放小一点，",
            "愤怒": "你会生气是可以理解的，先别急着在情绪最高的时候回应，",
            "积极": "这个状态挺值得保留下来。可以趁现在把有效经验记住，",
            "平静": "你现在的表达比较稳定，我们可以继续把事情梳理清楚，",
            "惊讶": "突然遇到这种信息，先有冲击感很正常，",
        }
        return openers.get(emotion.label, "我先接住你现在说的这件事，")

    def _reply_follow_up(self, emotion: EmotionResult, insight: EmotionInsight) -> str:
        if emotion.label in {"低落", "焦虑", "愤怒"}:
            return " 如果你愿意，可以继续告诉我最难受的是哪一部分，我再陪你往下拆。"
        if emotion.label == "积极":
            return " 你也可以告诉我这次顺利的关键点，我们一起把它变成可复用的方法。"
        return " 你可以继续说一个具体片段，我会根据上下文继续帮你整理。"

    def _build_summary(
        self,
        message: str,
        emotion: EmotionResult,
        trigger: str,
        history: list[ChatMessage],
        strategy: dict[str, object],
        review_mode: bool,
    ) -> str:
        context_note = "已有上下文" if history else "首轮表达"
        base_summary = str(strategy["summary_template"])

        if review_mode:
            return (
                f"{context_note}中，用户围绕“{trigger}”表达了{emotion.label}情绪"
                f"（指数 {emotion.score}/100）。RAG 检索到“{strategy['title']}”策略："
                f"{base_summary}建议下一步：{strategy['next_action']}"
            )

        return (
            f"{context_note}中，RAG 检索到“{strategy['title']}”策略。"
            f"{base_summary}"
        )
