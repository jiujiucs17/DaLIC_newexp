#!/usr/bin/env python3

import argparse
import json
import re
import unicodedata
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "rafaelkallis/tickettagger-roberta"


function_sig_regex = re.compile(r"[a-zA-Z][a-zA-Z0-9_.]*\([a-zA-Z0-9_, ]*\)")
issue_id_regex = re.compile(r"#[0-9]+")
non_ascii_char_regex = re.compile(r"[^\x00-\x7f]")
punctuations = "!\'\"`$%&\\()*,/:;<=>[\\]^{|}~+#@-_"
punctuations_trans = str.maketrans(punctuations, " " * len(punctuations))
whitespace_regex = re.compile(r"\s+")
TITLE_MAX_TOKENS = 20
MODEL_INPUT_TOKEN_BUDGET = 512


def get_repo_root() -> Path:
    # 不写死脚本所在层级，而是向上查找仓库根目录。
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()
    default_input = repo_root / "dataset" / "swebench.jsonl"
    default_output = repo_root / "dataset" / "swebench_nlbse23_roberta.jsonl"
    default_feature_output = repo_root / "dataset" / "swebench_feature.jsonl"

    parser = argparse.ArgumentParser(
        description="Classify SWE-bench instances with the official NLBSE'23 RoBERTa baseline."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--feature-output", type=Path, default=default_feature_output)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--export-feature-only",
        action="store_true",
        help="Skip classification and only export feature instances from an existing classification file.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.feature_output = args.feature_output.expanduser().resolve()

    if not args.input.exists():
        raise ValueError(f"Input file does not exist: {args.input}")
    if not args.input.is_file():
        raise ValueError(f"Input path is not a file: {args.input}")
    if args.input.suffix != ".jsonl":
        raise ValueError(f"Input file must be a .jsonl file: {args.input}")

    if args.output == args.input:
        raise ValueError("Output file must be different from input file.")
    if args.output.suffix != ".jsonl":
        raise ValueError(f"Output file must be a .jsonl file: {args.output}")
    if args.feature_output.suffix != ".jsonl":
        raise ValueError(f"Feature output file must be a .jsonl file: {args.feature_output}")
    if args.feature_output in {args.input, args.output}:
        raise ValueError("Feature output file must be different from both input and output files.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be a positive integer.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer when provided.")

    if args.export_feature_only and not args.output.exists():
        raise ValueError(f"Classification file does not exist: {args.output}")

def select_device() -> torch.device:
    # Apple Silicon 上优先使用 MPS；没有时再回退。
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess(text: str, max_tokens: int | None = None) -> str:
    text = str(text)
    text = function_sig_regex.sub(" FUNCTION ", text)
    text = issue_id_regex.sub(" ISSUE ", text)
    text = text.translate(punctuations_trans)
    text = non_ascii_char_regex.sub("", text)
    text = unicodedata.normalize("NFD", text)
    text = whitespace_regex.sub(" ", text).strip()
    if max_tokens is not None:
        text = " ".join(text.split()[:max_tokens])
    return text


def count_tokens(text: str) -> int:
    return len(str(text).split())


def split_problem_statement(problem_statement: str) -> tuple[str, str]:
    # 这里统一按“标题首行 + 正文其余行”切分，适用于当前规范化后的 SWE-bench 输入。
    parts = str(problem_statement).splitlines()
    if not parts:
        return "", ""
    title = parts[0]
    body = "\n".join(parts[1:]) if len(parts) > 1 else ""
    return title, body


def build_model_input(problem_statement: str) -> str:
    title, body = split_problem_statement(problem_statement)
    title_text = preprocess(title, max_tokens=TITLE_MAX_TOKENS)
    if title_text:
        model_input = f"TITLE {title_text}"
        body_token_budget = max(0, MODEL_INPUT_TOKEN_BUDGET - 1 - count_tokens(model_input))
        body_text = preprocess(body, max_tokens=body_token_budget)
        if body_text:
            model_input += f" BODY {body_text}"
        elif body:
            model_input += " BODY"

        if count_tokens(model_input) > MODEL_INPUT_TOKEN_BUDGET:
            raise ValueError("Constructed model input exceeds token budget before tokenization.")
        return model_input

    body_text = preprocess(body, max_tokens=MODEL_INPUT_TOKEN_BUDGET - 1)
    if body_text:
        return f"BODY {body_text}"
    return ""


def load_rows(path: Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            yield json.loads(line)


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def classify_rows(
    input_path: Path,
    output_path: Path,
    batch_size: int,
    max_length: int,
    limit: int | None,
) -> None:
    device = select_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    labels = get_labels_from_model(model)

    if max_length > tokenizer.model_max_length:
        raise ValueError(
            f"--max-length ({max_length}) exceeds tokenizer.model_max_length "
            f"({tokenizer.model_max_length})."
        )

    rows = load_rows(input_path, limit=limit)
    with input_path.open("r", encoding="utf-8") as input_f:
        total = limit if limit is not None else sum(1 for _ in input_f)
    if total == 0:
        raise ValueError(f"Input file is empty: {input_path}")
    rows = load_rows(input_path, limit=limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as temp_f:
            temp_path = Path(temp_f.name)
            for batch in tqdm(batched(rows, batch_size), total=(total + batch_size - 1) // batch_size):
                docs = [build_model_input(get_required_field(row, "problem_statement")) for row in batch]
                inputs = tokenizer(
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred_ids = logits.argmax(dim=-1).tolist()

                for row, pred_id in zip(batch, pred_ids):
                    label = labels[pred_id]
                    result = {
                        "instance_id": get_required_field(row, "instance_id"),
                        "class": label,
                    }
                    temp_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        temp_path.replace(output_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def get_required_field(row: dict, field_name: str) -> str:
    if field_name not in row:
        raise ValueError(f"Missing required field '{field_name}' in input row: {row}")
    return row[field_name]


def get_labels_from_model(model: AutoModelForSequenceClassification) -> list[str]:
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        raise ValueError("Model config does not contain id2label; cannot map predictions safely.")

    try:
        return [id2label[str(i)] if str(i) in id2label else id2label[i] for i in range(len(id2label))]
    except Exception as exc:
        raise ValueError(f"Failed to parse model id2label mapping: {id2label}") from exc


def load_feature_instance_ids(classification_path: Path) -> set[str]:
    feature_ids: set[str] = set()
    seen_ids: set[str] = set()
    for row in load_rows(classification_path):
        instance_id = get_required_field(row, "instance_id")
        if instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id found in classification file: {instance_id}")
        seen_ids.add(instance_id)
        if get_required_field(row, "class") == "feature":
            feature_ids.add(instance_id)

    if not feature_ids:
        raise ValueError(f"No feature instances found in classification file: {classification_path}")
    return feature_ids


def export_feature_rows(
    classification_path: Path,
    source_path: Path,
    feature_output_path: Path,
) -> None:
    feature_ids = load_feature_instance_ids(classification_path)
    matched_ids: set[str] = set()

    feature_output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=feature_output_path.parent,
            delete=False,
        ) as temp_f:
            temp_path = Path(temp_f.name)
            for row in load_rows(source_path):
                instance_id = get_required_field(row, "instance_id")
                if instance_id in feature_ids:
                    temp_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    matched_ids.add(instance_id)

        missing_ids = feature_ids - matched_ids
        if missing_ids:
            preview = ", ".join(sorted(missing_ids)[:5])
            raise ValueError(
                f"Could not find {len(missing_ids)} feature instance(s) in source data. "
                f"Examples: {preview}"
            )

        temp_path.replace(feature_output_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def main() -> None:
    args = parse_args()
    if not args.export_feature_only:
        classify_rows(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            max_length=args.max_length,
            limit=args.limit,
        )

    export_feature_rows(
        classification_path=args.output,
        source_path=args.input,
        feature_output_path=args.feature_output,
    )


if __name__ == "__main__":
    main()
