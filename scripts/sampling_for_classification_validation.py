#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


def get_repo_root() -> Path:
    for candidate in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (candidate / "dataset").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root from script path.")


def resolve_default_classification_input(repo_root: Path) -> Path:
    return repo_root / "dataset" / "swebench_nlbse23_roberta.jsonl"


def parse_args() -> argparse.Namespace:
    repo_root = get_repo_root()

    parser = argparse.ArgumentParser(
        description="Sample feature-classified SWE-bench instances for manual validation."
    )
    parser.add_argument(
        "--classification-input",
        type=Path,
        default=resolve_default_classification_input(repo_root),
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
    parser.add_argument("--sample-count", type=int, default=148)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
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
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be a positive integer.")


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            yield json.loads(line)


def get_required_field(row: dict, field_name: str):
    if field_name not in row:
        raise ValueError(f"Missing required field '{field_name}' in row: {row}")
    return row[field_name]


def load_classification_map(path: Path) -> dict[str, str]:
    label_by_id: dict[str, str] = {}
    for row in load_rows(path):
        instance_id = get_required_field(row, "instance_id")
        label = get_required_field(row, "class")
        if instance_id in label_by_id:
            raise ValueError(f"Duplicate instance_id found in classification file: {instance_id}")
        label_by_id[instance_id] = label

    if not label_by_id:
        raise ValueError(f"Classification file is empty: {path}")
    return label_by_id


def sample_instance_ids(
    label_by_id: dict[str, str],
    sample_count: int,
    seed: int,
) -> list[str]:
    feature_ids = [instance_id for instance_id, label in label_by_id.items() if label == "feature"]

    if len(feature_ids) < sample_count:
        raise ValueError(
            f"Not enough feature instances to sample {sample_count}. "
            f"Only found {len(feature_ids)}."
        )

    rng = random.Random(seed)
    return rng.sample(feature_ids, sample_count)


def load_source_map(path: Path) -> dict[str, dict]:
    source_by_id: dict[str, dict] = {}
    for row in load_rows(path):
        instance_id = get_required_field(row, "instance_id")
        if instance_id in source_by_id:
            raise ValueError(f"Duplicate instance_id found in source file: {instance_id}")
        source_by_id[instance_id] = row

    if not source_by_id:
        raise ValueError(f"Source file is empty: {path}")
    return source_by_id


def build_sample_rows(
    sampled_ids: list[str],
    label_by_id: dict[str, str],
    source_by_id: dict[str, dict],
) -> list[dict[str, str]]:
    sample_rows: list[dict[str, str]] = []
    missing_ids: list[str] = []

    for instance_id in sampled_ids:
        source_row = source_by_id.get(instance_id)
        if source_row is None:
            missing_ids.append(instance_id)
            continue

        sample_rows.append(
            {
                "instance_id": instance_id,
                "problem_statement": get_required_field(source_row, "problem_statement"),
                "class": label_by_id[instance_id],
            }
        )

    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"Could not find {len(missing_ids)} sampled instance(s) in source file. "
            f"Examples: {preview}"
        )
    return sample_rows


def column_name(index: int) -> str:
    name = ""
    current = index
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def make_inline_string_cell(cell_ref: str, value: str) -> str:
    escaped = escape("" if value is None else str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{escaped}</t></is></c>'


def build_sheet_xml(headers: list[str], rows: list[dict[str, str]]) -> str:
    row_xml_parts: list[str] = []

    header_cells = []
    for index, header in enumerate(headers, start=1):
        header_cells.append(make_inline_string_cell(f"{column_name(index)}1", header))
    row_xml_parts.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_index, row in enumerate(rows, start=2):
        cells = []
        for col_index, header in enumerate(headers, start=1):
            cell_ref = f"{column_name(col_index)}{row_index}"
            cells.append(make_inline_string_cell(cell_ref, row.get(header, "")))
        row_xml_parts.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    rows_xml = "".join(row_xml_parts)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{rows_xml}</sheetData>"
        "</worksheet>"
    )


def write_xlsx(output_path: Path, rows: list[dict[str, str]]) -> None:
    headers = ["instance_id", "problem_statement", "class"]
    sheet_xml = build_sheet_xml(headers, rows)
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
        "</Types>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="sample" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
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
        "</Relationships>"
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
        "</styleSheet>"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

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


def main() -> None:
    args = parse_args()
    label_by_id = load_classification_map(args.classification_input)
    sampled_ids = sample_instance_ids(
        label_by_id=label_by_id,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    source_by_id = load_source_map(args.source_input)
    sample_rows = build_sample_rows(
        sampled_ids=sampled_ids,
        label_by_id=label_by_id,
        source_by_id=source_by_id,
    )
    write_xlsx(args.output, sample_rows)


if __name__ == "__main__":
    main()
