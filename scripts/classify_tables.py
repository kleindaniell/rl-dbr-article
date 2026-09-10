#!/usr/bin/env python3
"""Classify the 23 pandoc-JSON Table blocks extracted from the manuscript
into: equation-layout, figure-layout, or real data table. Read-only report,
makes no changes."""
import json
import sys


def inline_text(node):
    if isinstance(node, list):
        return "".join(inline_text(n) for n in node)
    if not isinstance(node, dict):
        return ""
    t = node.get("t")
    if t == "Str":
        return node["c"]
    if t == "Space":
        return " "
    if t == "SoftBreak":
        return " "
    if t in ("Emph", "Strong", "Span", "Underline"):
        c = node["c"]
        inner = c[-1] if isinstance(c, list) and c and isinstance(c[-1], list) else c
        return inline_text(inner)
    if t == "Image":
        return "[IMAGE]"
    if t == "Math":
        return "[MATH:%s]" % node["c"][1]
    if "c" in node:
        return inline_text(node["c"])
    return ""


def cell_text(cell):
    # cell = [Attr, Alignment, RowSpan, ColSpan, [Block]]
    blocks = cell[4]
    parts = []
    for b in blocks:
        if b["t"] == "Para" or b["t"] == "Plain":
            parts.append(inline_text(b["c"]))
    return " ".join(parts).strip()


def row_cells(row):
    # row = [Attr, [Cell]]
    return row[1]


def main():
    doc = json.load(open("markdown/manuscript.json"))
    blocks = doc["blocks"]
    idx = 0
    for bi, b in enumerate(blocks):
        if b["t"] != "Table":
            continue
        idx += 1
        c = b["c"]
        # Table = [Attr, Caption, [ColSpec], TableHead, [TableBody], TableFoot]
        caption_blocks = c[1][1]
        caption = " ".join(inline_text(cb["c"]) for cb in caption_blocks if "c" in cb)
        head = c[3]
        bodies = c[4]
        head_rows = head[1]
        body_rows = [r for body in bodies for r in body[3]]
        all_rows = head_rows + body_rows
        nrows = len(all_rows)
        ncols = len(c[2])
        texts = []
        for r in all_rows:
            for cell in row_cells(r):
                texts.append(cell_text(cell))
        joined = " | ".join(texts)

        import re
        has_image = any("[IMAGE]" in t for t in texts)

        def cell_kind(t):
            s = t.strip()
            if s == "":
                return "EMPTY"
            if re.fullmatch(r"\(\d{1,2}\)", s):
                return "NUM"
            if s.count("[MATH:") == 1:
                stripped = re.sub(r"\[MATH:.*\]", "", s, flags=re.DOTALL).strip()
                if stripped.strip(",.;: ") == "":
                    return "MATHCELL"
            return "OTHER"

        kinds = [cell_kind(t) for t in texts]
        is_eqnum = (
            not has_image
            and nrows <= 3
            and all(k in ("EMPTY", "NUM", "MATHCELL") for k in kinds)
            and "NUM" in kinds
            and "MATHCELL" in kinds
        )
        label = "DATA"
        if is_eqnum:
            label = "EQNUM"
        elif has_image:
            label = "FIGLAYOUT"

        print(f"--- Table #{idx} (block {bi}) rows={nrows} cols={ncols} label={label} ---")
        if caption:
            print(f"  caption: {caption[:120]}")
        preview = joined[:200].replace("\n", " ")
        print(f"  content: {preview}")
        print()


if __name__ == "__main__":
    main()
