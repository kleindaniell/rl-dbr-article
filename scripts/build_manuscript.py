#!/usr/bin/env python3
"""Transform the raw pandoc JSON AST (markdown/manuscript.json) into a
structurally-cleaned AST (markdown/manuscript.cleaned.json), applying the
Phase 3 fixes from the project plan:

  - 18 equation-numbering layout tables -> real numbered `equation` blocks
    (LaTeX, tagged with their original number).
  - 2 figure sub-panel layout tables (Figures 6 and 7) -> LaTeX subfigure
    groups.
  - The duplicated Section 4.3 ("[REVISED DRAFT]" / "ORIGINAL TEXT BELOW")
    is kept in full, both halves visibly flagged, not resolved.
  - The second colliding "Table 2." caption is renumbered to "Table 3."
  - Figure image paths are repointed from word/media/imageN.png to the
    renamed files in figures/.

Read-only on the source docx; writes only markdown/manuscript.cleaned.json.
"""
import json
import re
import subprocess

IMAGE_MAP = {
    "media/image1.png": "figures/fig01.png",
    "media/image2.png": "figures/fig02.png",
    "media/image3.png": "figures/fig03.png",
    "media/image4.png": "figures/fig04.png",
    "media/image5.png": "figures/fig05.png",
    "media/image6.png": "figures/fig06a.png",
    "media/image7.png": "figures/fig06bcde.png",
    "media/image8.png": "figures/fig07abcd.png",
    "media/image9.png": "figures/fig08.png",
    "media/image10.png": "figures/fig09.png",
    "media/image11.png": "figures/fig10.png",
}

PANDOC_API_VERSION = [1, 23, 1, 2]


def inline_text(node):
    if isinstance(node, list):
        return "".join(inline_text(n) for n in node)
    if not isinstance(node, dict):
        return ""
    t = node.get("t")
    if t == "Str":
        return node["c"]
    if t in ("Space", "SoftBreak"):
        return " "
    if t == "Math":
        return "[MATH]"
    if t == "Image":
        return "[IMAGE]"
    if "c" in node:
        return inline_text(node["c"])
    return ""


def inlines_to_latex(inlines):
    """Render a list of pandoc Inline nodes to LaTeX via pandoc itself, so
    italics/special characters/symbols are handled correctly."""
    doc = {
        "pandoc-api-version": PANDOC_API_VERSION,
        "meta": {},
        "blocks": [{"t": "Para", "c": inlines}],
    }
    p = subprocess.run(
        ["pandoc", "-f", "json", "-t", "latex"],
        input=json.dumps(doc),
        capture_output=True,
        text=True,
        check=True,
    )
    return p.stdout.strip()


def rewrite_image_paths(node):
    if isinstance(node, list):
        for n in node:
            rewrite_image_paths(n)
    elif isinstance(node, dict):
        if node.get("t") == "Image":
            target = node["c"][2]
            if target[0] in IMAGE_MAP:
                target[0] = IMAGE_MAP[target[0]]
        for v in node.values():
            rewrite_image_paths(v)


def raw_latex(s):
    return {"t": "RawBlock", "c": ["latex", s]}


def table_rows(table_block):
    c = table_block["c"]
    head_rows = c[3][1]
    body_rows = [r for body in c[4] for r in body[3]]
    return head_rows + body_rows


def cell_blocks(cell):
    return cell[4]


def cell_text(cell):
    parts = []
    for b in cell_blocks(cell):
        if b["t"] in ("Para", "Plain"):
            parts.append(inline_text(b["c"]))
    return " ".join(parts).strip()


def cell_math_and_inlines(cell):
    """Return (latex_math_string, full_inline_list) if this cell contains
    exactly one Math inline and nothing else meaningful, else (None, None)."""
    math_nodes = []

    def walk(n):
        if isinstance(n, list):
            for x in n:
                walk(x)
        elif isinstance(n, dict):
            if n.get("t") == "Math":
                math_nodes.append(n)
            else:
                for v in n.values():
                    walk(v)

    walk(cell_blocks(cell))
    if len(math_nodes) != 1:
        return None
    return math_nodes[0]["c"][1]


def cell_images(cell):
    imgs = []

    def walk(n):
        if isinstance(n, list):
            for x in n:
                walk(x)
        elif isinstance(n, dict):
            if n.get("t") == "Image":
                imgs.append(n)
            else:
                for v in n.values():
                    walk(v)

    walk(cell_blocks(cell))
    return imgs


def classify_table(b):
    rows = table_rows(b)
    nrows = len(rows)
    texts = [cell_text(cell) for r in rows for cell in r[1]]
    has_image = any(cell_images(cell) for r in rows for cell in r[1])

    def cell_kind(t):
        s = t.strip()
        if s == "":
            return "EMPTY"
        if re.fullmatch(r"\(\d{1,2}\)", s):
            return "NUM"
        if s.count("[MATH:") == 0 and re.search(r"\\[a-zA-Z]", ""):
            pass
        return "OTHER"

    # classify using cell_math_and_inlines for MATHCELL detection
    kinds = []
    for r in rows:
        for cell in r[1]:
            t = cell_text(cell)
            s = t.strip()
            if s == "":
                kinds.append("EMPTY")
            elif re.fullmatch(r"\(\d{1,2}\)", s):
                kinds.append("NUM")
            else:
                math = cell_math_and_inlines(cell)
                if math is not None:
                    stripped = re.sub(r"\$.*\$", "", s)  # not used, kept simple
                    kinds.append("MATHCELL")
                else:
                    kinds.append("OTHER")

    if not has_image and nrows <= 3 and "NUM" in kinds and "MATHCELL" in kinds \
            and all(k in ("EMPTY", "NUM", "MATHCELL") for k in kinds):
        return "EQNUM"
    if has_image:
        return "FIGLAYOUT"
    return "DATA"


def equations_from_table(b):
    out = []
    for r in table_rows(b):
        cells = r[1]
        maths = []
        nums = []
        for cell in cells:
            t = cell_text(cell).strip()
            if re.fullmatch(r"\(\d{1,2}\)", t):
                nums.append(t.strip("()"))
            else:
                math = cell_math_and_inlines(cell)
                if math is not None:
                    maths.append(math)
        for math, num in zip(maths, nums):
            out.append(raw_latex(
                "\\begin{equation}\n%s\n\\tag{%s}\n\\end{equation}" % (math, num)
            ))
    return out


def subfigure_from_table(b, caption_inlines):
    rows = table_rows(b)
    # rows come in (image-row, label-row) pairs
    parts = [r"\begin{figure}[htbp]", r"\centering"]
    letter_to_img = {}
    i = 0
    while i < len(rows):
        img_cells = rows[i][1]
        label_cells = rows[i + 1][1] if i + 1 < len(rows) else []
        ncols = max(len(img_cells), 1)
        width = r"\textwidth" if len(img_cells) == 1 else r"0.48\textwidth"
        for j, cell in enumerate(img_cells):
            imgs = cell_images(cell)
            label = cell_text(label_cells[j]) if j < len(label_cells) else ""
            label = label.strip("() ") or "?"
            if imgs:
                src = imgs[0]["c"][2][0]
                parts.append(
                    r"\begin{subfigure}[b]{%s}" % width
                )
                parts.append(r"\centering")
                parts.append(r"\includegraphics[width=\linewidth]{%s}" % src)
                parts.append(r"\caption{}")
                parts.append(r"\label{fig:sub:%s}" % label)
                parts.append(r"\end{subfigure}")
        i += 2
        parts.append(r"\hfill" if False else "%")
    cap_latex = inlines_to_latex(caption_inlines)
    parts.append(r"\caption{%s}" % cap_latex)
    parts.append(r"\end{figure}")
    return raw_latex("\n".join(parts))


def fix_table2_collision(blocks):
    seen = False
    for b in blocks:
        if b["t"] in ("Para", "Plain"):
            txt = inline_text(b["c"])
            if txt.strip().startswith("Table 2."):
                if seen:
                    for node in b["c"]:
                        if node.get("t") == "Str" and node["c"] == "2.":
                            node["c"] = "3."
                            break
                else:
                    seen = True


def fix_header_levels(blocks):
    """All headings use one flat Word style ('Ttulo1'), so pandoc assigns
    them all level 1. Recompute the real level from the section numbering
    (e.g. '2.1 ...' -> level 2) so \\section/\\subsection nest correctly.
    Also strip the leading number from the heading text itself, since
    LaTeX's \\section/\\subsection will supply its own numbering -- leaving
    the source number in would double it up ("2 2 Theoretical Background")."""
    for b in blocks:
        if b["t"] != "Header":
            continue
        inlines = b["c"][2]
        txt = inline_text(inlines)
        m = re.match(r"^(\d+)\.(\d+)\b", txt)
        if m:
            b["c"][0] = 2
        else:
            m1 = re.match(r"^(\d+)\.?\s", txt)
            if m1:
                b["c"][0] = 1
        if inlines and inlines[0].get("t") == "Str" and \
                re.fullmatch(r"\d+(\.\d+)?\.?", inlines[0]["c"]):
            del inlines[0]
            if inlines and inlines[0].get("t") == "Space":
                del inlines[0]


CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_control_chars(node):
    if isinstance(node, list):
        for n in node:
            strip_control_chars(n)
    elif isinstance(node, dict):
        if node.get("t") == "Str" and isinstance(node.get("c"), str):
            node["c"] = CONTROL_CHAR_RE.sub("", node["c"])
        for v in node.values():
            strip_control_chars(v)


def main():
    doc = json.load(open("markdown/manuscript.json"))
    blocks = doc["blocks"]
    strip_control_chars(blocks)
    rewrite_image_paths(blocks)
    fix_header_levels(blocks)

    new_blocks = []
    skip = set()
    for i, b in enumerate(blocks):
        if i in skip:
            continue
        t = b["t"]

        if t == "Header":
            txt = inline_text(b["c"][2])
            if "ORIGINAL TEXT BELOW" in txt:
                new_blocks.append(raw_latex(
                    r"\bigskip\noindent\hrulefill\par\smallskip"
                    r"\noindent\textbf{Original version} "
                    r"(kept for comparison --- the revised draft above and "
                    r"this original section have not yet been reconciled "
                    r"by the author).\par\bigskip"
                ))
                continue
            new_blocks.append(b)
            if "[REVISED DRAFT" in txt:
                new_blocks.append(raw_latex(
                    r"\noindent\colorbox{yellow!40}{REVISED DRAFT --- pending author review. "
                    r"Kept alongside the original version of this section "
                    r"below; the two have not yet been reconciled (see the "
                    r"reviewer note at the end of this block).}\par\medskip"
                ))
            continue

        if t == "Table":
            kind = classify_table(b)
            if kind == "EQNUM":
                new_blocks.extend(equations_from_table(b))
                continue
            if kind == "FIGLAYOUT":
                cap_block = blocks[i + 1]
                skip.add(i + 1)
                new_blocks.append(subfigure_from_table(b, cap_block["c"]))
                continue
            new_blocks.append(b)  # DATA table, unchanged
            continue

        new_blocks.append(b)

    fix_table2_collision(new_blocks)

    doc["blocks"] = new_blocks
    json.dump(doc, open("markdown/manuscript.cleaned.json", "w"))
    print(f"wrote markdown/manuscript.cleaned.json ({len(new_blocks)} blocks)")


if __name__ == "__main__":
    main()
