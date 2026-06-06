import argparse
import json
import os
import random
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
HF_CACHE_DIR = BACKEND_DIR / "models" / "hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from app.ml.bert_sentiment import ID_TO_LABEL, LABEL_TO_ID, MODEL_NAME, SentimentClassifier, SentimentDataset

DEFAULT_DATA_PATH = Path(r"D:\java\vibecoding\python-backend\data\train.csv")
DEFAULT_TIX007_TRAIN_PATH = BACKEND_DIR / "data" / "emotion_tix007" / "data" / "train.csv"
DEFAULT_TIX007_VALIDATION_PATH = BACKEND_DIR / "data" / "emotion_tix007" / "data" / "validation.csv"
DEFAULT_SMP_DIR = BACKEND_DIR / "external" / "BERT_SMP2020-EWECT" / "data" / "clean"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "models" / "sentiment"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[["text", "label"]].dropna()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"] != ""]

    invalid_labels = sorted(set(frame["label"].unique()) - set(LABEL_TO_ID))
    if invalid_labels:
        raise ValueError(f"Invalid labels: {invalid_labels}")

    frame["label_id"] = frame["label"].map(LABEL_TO_ID)
    return frame


def load_training_frame(data_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    required_columns = {"text", "label"}

    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Training data must contain columns: {sorted(required_columns)}")

    frame["label"] = frame["label"].map({"正面": "快乐", "负面": "悲伤", "中性": "平静"})
    return normalize_frame(frame)


def load_tix007_frame(data_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    required_columns = {"text", "label_cn"}

    if not required_columns.issubset(frame.columns):
        raise ValueError(f"TIX007 data must contain columns: {sorted(required_columns)}")

    frame = frame[["text", "label_cn"]].rename(columns={"label_cn": "label"})
    return normalize_frame(frame)


def load_smp_frame(data_paths: list[Path]) -> pd.DataFrame:
    import json

    label_map = {
        "happy": "快乐",
        "angry": "愤怒",
        "sad": "悲伤",
        "fear": "恐惧",
        "surprise": "惊讶",
        "neutral": "平静",
    }
    records = []

    for data_path in data_paths:
        with data_path.open("r", encoding="utf-8") as file:
            items = json.load(file)

        for item in items:
            records.append(
                {
                    "text": item.get("content", ""),
                    "label": label_map.get(item.get("label")),
                }
            )

    return normalize_frame(pd.DataFrame(records))


def load_combined_frame(args: argparse.Namespace) -> pd.DataFrame:
    frames = []

    if args.include_weibo:
        frames.append(load_training_frame(Path(args.data)))

    if args.include_tix007:
        frames.append(load_tix007_frame(Path(args.tix007_train)))

    if args.include_smp:
        frames.append(
            load_smp_frame(
                [
                    Path(args.smp_dir) / "usual_train.txt",
                    Path(args.smp_dir) / "virus_train.txt",
                ]
            )
        )

    if not frames:
        raise ValueError("At least one data source must be enabled.")

    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)


def evaluate(model: SentimentClassifier, loader: DataLoader, device: torch.device) -> dict[str, float | str]:
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs["logits"], dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(all_labels, all_predictions, average="macro"),
        "report": classification_report(
            all_labels,
            all_predictions,
            labels=sorted(ID_TO_LABEL),
            target_names=[ID_TO_LABEL[index] for index in sorted(ID_TO_LABEL)],
            digits=4,
            zero_division=0,
        ),
    }


def save_model(
    output_dir: Path,
    model: SentimentClassifier,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    metrics: dict[str, float | str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "model_name": MODEL_NAME,
                "num_labels": len(LABEL_TO_ID),
                "max_length": args.max_length,
                "label_to_id": LABEL_TO_ID,
                "id_to_label": ID_TO_LABEL,
            },
            "metrics": metrics,
        },
        output_dir / "model.pt",
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def train(args: argparse.Namespace) -> dict[str, float | str]:
    set_seed(args.seed)

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    frame = load_combined_frame(args)

    if args.max_per_label > 0:
        frame = (
            frame.groupby("label_id", group_keys=False)
            .apply(lambda group: group.sample(n=min(len(group), args.max_per_label), random_state=args.seed))
            .sample(frac=1, random_state=args.seed)
            .reset_index(drop=True)
        )

    if args.max_samples and args.max_samples < len(frame):
        frame = frame.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    train_frame, validation_frame = train_test_split(
        frame,
        test_size=args.validation_ratio,
        random_state=args.seed,
        stratify=frame["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(HF_CACHE_DIR))
    train_dataset = SentimentDataset(
        train_frame["text"].tolist(),
        train_frame["label_id"].tolist(),
        tokenizer,
        args.max_length,
    )
    validation_dataset = SentimentDataset(
        validation_frame["text"].tolist(),
        validation_frame["label_id"].tolist(),
        tokenizer,
        args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SentimentClassifier(model_name=MODEL_NAME, num_labels=len(LABEL_TO_ID), freeze_bert=args.freeze_bert)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(total_steps // 10, 1)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0

        for step, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            if step % args.log_steps == 0:
                progress.set_postfix(loss=f"{running_loss / step:.4f}")

    metrics = evaluate(model, validation_loader, device)
    save_model(output_dir, model, tokenizer, args, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT sentiment model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--tix007-train", default=str(DEFAULT_TIX007_TRAIN_PATH))
    parser.add_argument("--smp-dir", default=str(DEFAULT_SMP_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--include-weibo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-tix007", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-smp", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    metrics = train(parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
