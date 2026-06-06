import os
import re
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = BACKEND_DIR / "models" / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
LABEL_TO_ID = {"悲伤": 0, "快乐": 1, "爱": 2, "愤怒": 3, "恐惧": 4, "惊讶": 5, "平静": 6}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def preprocess_text(text: str) -> str:
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", str(text))
    return re.sub(r"\s+", " ", text).strip()


class SentimentDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = preprocess_text(self.texts[index])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class SentimentClassifier(nn.Module):
    """Transformer sentiment classifier matching mg1094 model head design."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = 3,
        freeze_bert: bool = False,
        hidden_dropout_prob: float = 0.1,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(
            model_name,
            return_dict=True,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=local_files_only,
        )

        if freeze_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self._init_weights(self.classifier)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.last_hidden_state,
        }
