from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "team_project_guide.md"
OUTPUT = ROOT / "docs" / "team_project_guide.pdf"


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _clean_markdown(line: str) -> tuple[str, int]:
    line = line.rstrip()
    if line.startswith("# "):
        return line[2:], 18
    if line.startswith("## "):
        return line[3:], 15
    if line.startswith("### "):
        return line[4:], 13
    if line.startswith("- "):
        return "- " + line[2:], 10
    if line.startswith("> "):
        return line[2:], 10
    if line.startswith("```"):
        return "", 10
    line = re.sub(r"`([^`]+)`", r"\1", line)
    line = line.replace("**", "")
    return line, 10


def _lines_from_markdown(markdown: str) -> list[tuple[str, int]]:
    output: list[tuple[str, int]] = []
    in_code = False
    for raw_line in markdown.splitlines():
        if raw_line.startswith("```"):
            in_code = not in_code
            output.append(("", 10))
            continue
        if in_code:
            wrapped = textwrap.wrap(raw_line, width=88) or [""]
            output.extend((line, 9) for line in wrapped)
            continue
        text, size = _clean_markdown(raw_line)
        if not text:
            output.append(("", 10))
            continue
        width = 72 if size <= 10 else 56
        wrapped = textwrap.wrap(text, width=width, replace_whitespace=False) or [""]
        output.extend((line, size) for line in wrapped)
    return output


def _paginate(lines: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
    pages: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    y = 760
    for text, size in lines:
        step = size + 6
        if y - step < 60:
            pages.append(current)
            current = []
            y = 760
        current.append((text, size))
        y -= step
    if current:
        pages.append(current)
    return pages


def _build_page_stream(page: list[tuple[str, int]], page_number: int) -> str:
    commands = ["BT"]
    current_y = 760
    for text, size in page:
        commands.append(f"/F1 {size} Tf")
        safe = _escape_pdf_text(text)
        commands.append(f"1 0 0 1 50 {current_y} Tm")
        commands.append(f"({safe}) Tj")
        current_y -= size + 6
    commands.append("/F1 9 Tf")
    commands.append("1 0 0 1 50 35 Tm")
    commands.append(f"(GPU Optimizer Triton - Guia del equipo | Pagina {page_number}) Tj")
    commands.append("ET")
    return "\n".join(commands)


def write_pdf(pages: list[list[tuple[str, int]]], output: Path) -> None:
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    page_count = len(pages)
    page_object_ids = [3 + index * 2 for index in range(page_count)]
    kids = " ".join(f"{object_id} 0 R" for object_id in page_object_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("latin-1"))

    for index, page in enumerate(pages, start=1):
        page_id = 3 + (index - 1) * 2
        content_id = page_id + 1
        stream = _build_page_stream(page, index).encode("latin-1", errors="replace")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    output.write_bytes(pdf)


def main() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    pages = _paginate(_lines_from_markdown(markdown))
    write_pdf(pages, OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
