#!/usr/bin/env python3

# accepted arguments:
# --classification-input: path to the .jsonl file containing the classification results (default: dataset/swebench_nlbse23_roberta.jsonl)
# --source-input: path to the .jsonl file containing the original problem statements (default: dataset/swebench.jsonl)
# --output: path to the output .xlsx file (default: dataset/swebench_sample.xlsx)
# --num-feature: number of "feature" samples to include (default: 148)
# --num-non-feature: number of "non-feature" samples to include (default: 0)
# --seed: random seed for sampling (default: 42)


import argparse
import json
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

HEADERS = ["instance_id", "problem_statement", "class"]


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()
    parser = argparse.ArgumentParser(
        description="Sample classified SWE-bench instances for manual validation."
    )
    parser.add_argument(
        "--classification-input",
        type=Path,
        default=repo_root / "dataset" / "swebench_nlbse23_roberta.jsonl",
    )
    parser.add_argument(
        "--source-input",
        type=Path,
        default=repo_root / "dataset" / "swebench.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "dataset" / "swebench_sample.xlsx",
    )
    parser.add_argument("--num-feature", "--num_feature", dest="num_feature", type=int, default=148)
    parser.add_argument(
        "--num-non-feature",
        "--num_non_feature",
        dest="num_non_feature",
        type=int,
        default=0,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return validate_args(args)


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.classification_input = args.classification_input.expanduser().resolve()
    args.source_input = args.source_input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.classification_input.exists():
        raise ValueError(f"Classification file does not exist: {args.classification_input}")
    if not args.source_input.exists():
        raise ValueError(f"Source file does not exist: {args.source_input}")
    if args.classification_input.suffix != ".jsonl":
        raise ValueError("--classification-input must be a .jsonl file.")
    if args.source_input.suffix != ".jsonl":
        raise ValueError("--source-input must be a .jsonl file.")
    if args.output.suffix != ".xlsx":
        raise ValueError("--output must be a .xlsx file.")
    if args.num_feature < 0:
        raise ValueError("--num-feature must be a non-negative integer.")
    if args.num_non_feature < 0:
        raise ValueError("--num-non-feature must be a non-negative integer.")
    return args


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            yield json.loads(line)


def require_field(row: dict, field_name: str, path: Path):
    if field_name not in row:
        raise ValueError(f"Missing required field {field_name!r} in {path}: {row}")
    return row[field_name]


def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in load_jsonl(path):
        instance_id = require_field(row, "instance_id", path)
        label = require_field(row, "class", path)
        if instance_id in labels:
            raise ValueError(f"Duplicate instance_id found in classification file: {instance_id}")
        labels[instance_id] = label
    if not labels:
        raise ValueError(f"Classification file is empty: {path}")
    return labels


def load_problem_statements(path: Path) -> dict[str, str]:
    problem_statements: dict[str, str] = {}
    for row in load_jsonl(path):
        instance_id = require_field(row, "instance_id", path)
        problem_statement = require_field(row, "problem_statement", path)
        if instance_id in problem_statements:
            raise ValueError(f"Duplicate instance_id found in source file: {instance_id}")
        problem_statements[instance_id] = problem_statement
    if not problem_statements:
        raise ValueError(f"Source file is empty: {path}")
    return problem_statements


def sample_instance_ids(
    labels: dict[str, str],
    num_feature: int,
    num_non_feature: int,
    seed: int,
) -> list[str]:
    feature_ids = [instance_id for instance_id, label in labels.items() if label == "feature"]
    non_feature_ids = [instance_id for instance_id, label in labels.items() if label != "feature"]

    rng = random.Random(seed)
    sampled_ids = []

    if num_feature:
        if num_feature > len(feature_ids):
            raise ValueError(
                f"Requested {num_feature} feature samples, but only {len(feature_ids)} are available."
            )
        sampled_ids.extend(rng.sample(feature_ids, k=num_feature))

    if num_non_feature:
        if num_non_feature > len(non_feature_ids):
            raise ValueError(
                "Requested "
                f"{num_non_feature} non-feature samples, but only {len(non_feature_ids)} are available."
            )
        sampled_ids.extend(rng.sample(non_feature_ids, k=num_non_feature))

    rng.shuffle(sampled_ids)
    return sampled_ids


def build_sample_rows(
    sampled_ids: list[str],
    labels: dict[str, str],
    problem_statements: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for instance_id in sampled_ids:
        if instance_id not in problem_statements:
            raise ValueError(f"Sampled instance_id not found in source file: {instance_id}")
        rows.append(
            {
                "instance_id": instance_id,
                "problem_statement": problem_statements[instance_id],
                "class": labels[instance_id],
            }
        )
    return rows


def column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def make_inline_string_cell(cell_ref: str, value: str) -> str:
    escaped = escape("" if value is None else str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{escaped}</t></is></c>'


def build_sheet_xml(rows: list[dict[str, str]]) -> str:
    row_xml_parts = []
    header_cells = [
        make_inline_string_cell(f"{column_name(index)}1", header)
        for index, header in enumerate(HEADERS, start=1)
    ]
    row_xml_parts.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_index, row in enumerate(rows, start=2):
        cells = [
            make_inline_string_cell(f"{column_name(col_index)}{row_index}", row.get(header, ""))
            for col_index, header in enumerate(HEADERS, start=1)
        ]
        row_xml_parts.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml_parts)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_xml = build_sheet_xml(rows)

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        '</Types>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="sample" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        '</Relationships>'
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        '</styleSheet>'
    )

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=output_path.parent, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            with ZipFile(temp_file, "w", compression=ZIP_DEFLATED) as workbook_zip:
                workbook_zip.writestr("[Content_Types].xml", content_types_xml)
                workbook_zip.writestr("_rels/.rels", rels_xml)
                workbook_zip.writestr("xl/workbook.xml", workbook_xml)
                workbook_zip.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
                workbook_zip.writestr("xl/worksheets/sheet1.xml", sheet_xml)
                workbook_zip.writestr("xl/styles.xml", styles_xml)
        temp_path.replace(output_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def print_summary(output_path: Path, rows: list[dict[str, str]]) -> None:
    total_rows = len(rows)
    feature_rows = sum(row["class"] == "feature" for row in rows)
    unique_instance_ids = len({row["instance_id"] for row in rows})

    print(f"Wrote {total_rows} sampled rows to {output_path}")
    print(f"Feature rows: {feature_rows}")
    print(f"Non-feature rows: {total_rows - feature_rows}")
    print(f"Unique instance_ids: {unique_instance_ids}")


def main() -> None:
    args = parse_args()
    labels = load_labels(args.classification_input)
    problem_statements = load_problem_statements(args.source_input)
    sampled_ids = sample_instance_ids(
        labels=labels,
        num_feature=args.num_feature,
        num_non_feature=args.num_non_feature,
        seed=args.seed,
    )
    sample_rows = build_sample_rows(sampled_ids, labels, problem_statements)
    write_xlsx(args.output, sample_rows)
    print_summary(args.output, sample_rows)


if __name__ == "__main__":
    main()
