#!/usr/bin/env python3
"""Split markdown/manuscript.cleaned.json into per-section .tex files under
sections/, plus emit the title/abstract/keywords text for main.tex to use.
Stops at the References header (bibliography is handled by biblatex +
references.bib instead of the raw reference-list text)."""
import json
import re
import subprocess

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
    if "c" in node:
        return inline_text(node["c"])
    return ""


def blocks_to_latex(blocks):
    doc = {"pandoc-api-version": PANDOC_API_VERSION, "meta": {}, "blocks": blocks}
    p = subprocess.run(
        ["pandoc", "-f", "json", "-t", "latex"],
        input=json.dumps(doc), capture_output=True, text=True, check=True,
    )
    return p.stdout


def slugify(text):
    text = re.sub(r"^[\d.]+\s*", "", text)  # drop leading "4.3 " etc
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:40]


def main():
    doc = json.load(open("markdown/manuscript.cleaned.json"))
    blocks = doc["blocks"]

    title_text = inline_text(blocks[0]["c"])
    abstract_text = inline_text(blocks[1]["c"])
    abstract_text = re.sub(r"^Abstract:\s*", "", abstract_text)
    keywords_text = inline_text(blocks[2]["c"])
    keywords_text = re.sub(r"^Keywords:\s*", "", keywords_text)

    body = blocks[3:]

    sections = []  # list of (level, title, [blocks])
    cur_title, cur_level, cur_blocks = None, None, []
    stop = False
    for b in body:
        if stop:
            break
        if b["t"] == "Header":
            htxt = inline_text(b["c"][2])
            if htxt.strip() == "References":
                stop = True
                break
            if cur_title is not None:
                sections.append((cur_level, cur_title, cur_blocks))
            cur_level = b["c"][0]
            cur_title = htxt
            cur_blocks = [b]
        else:
            cur_blocks.append(b)
    if cur_title is not None:
        sections.append((cur_level, cur_title, cur_blocks))

    manifest = []
    n = 0
    for level, title, blks in sections:
        if level == 1:
            n += 1
            fname = f"{n:02d}-{slugify(title)}.tex"
        else:
            fname = f"{n:02d}-{slugify(title)}-sub.tex"
        latex = blocks_to_latex(blks)
        # soul's \hl{} breaks under XeTeX on Unicode/nested-hyperref content
        # (forces a fixed-width font mid-scan); colorbox has no such issue.
        latex = latex.replace(r"\hl{", r"\colorbox{yellow!40}{")
        with open(f"sections/{fname}", "w") as f:
            f.write(latex)
        manifest.append(fname)
        print(f"wrote sections/{fname}  (level {level}: {title[:60]})")

    with open("scripts/section_manifest.txt", "w") as f:
        f.write("\n".join(manifest))

    with open("scripts/frontmatter.json", "w") as f:
        json.dump({
            "title": title_text,
            "abstract": abstract_text,
            "keywords": keywords_text,
        }, f, indent=2)

    print(f"\n{len(manifest)} section files written; manifest -> scripts/section_manifest.txt")


if __name__ == "__main__":
    main()
