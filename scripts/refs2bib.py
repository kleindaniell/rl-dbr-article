#!/usr/bin/env python3
"""Parse the 54-entry Emerald/T&F-style Mendeley reference list (extracted
from the original docx AST) into references.bib. Regex-based, no network
calls (DOIs are already present in the source text). Known duplicates
(Atwater & Chakravorty 2002a/2002b; the two Cox/Schleier 2010 Theory of
Constraints Handbook entries) are merged. Entries that don't parse cleanly
fall back to an @misc entry carrying the raw text in a note, and are listed
as warnings so nothing is silently dropped or fabricated."""
import json
import re


def inline_marked(node):
    """Render inlines to plain text, wrapping Emph spans in @@...@@ markers
    so the (usually singular) italicized journal/book title can be
    located after the fact."""
    if isinstance(node, list):
        return "".join(inline_marked(n) for n in node)
    if not isinstance(node, dict):
        return ""
    t = node.get("t")
    if t == "Str":
        return node["c"]
    if t in ("Space", "SoftBreak"):
        return " "
    if t == "Emph":
        return "@@" + inline_marked(node["c"]) + "@@"
    if "c" in node:
        return inline_marked(node["c"])
    return ""


def get_reference_paragraphs():
    doc = json.load(open("markdown/manuscript.json"))
    blocks = doc["blocks"]
    start = None
    for i, b in enumerate(blocks):
        if b["t"] == "Header" and inline_marked(b["c"][2]).strip() == "References":
            start = i + 1
            break
    assert start is not None, "References header not found"
    out = []
    for b in blocks[start:]:
        if b["t"] in ("Para", "Plain"):
            txt = inline_marked(b["c"]).strip()
            if txt:
                out.append(re.sub(r"\s+", " ", txt))
    return out


def bibtex_escape(s):
    return s.replace("&", r"\&").replace("%", r"\%")


def make_key(authors_raw, year):
    first_surname = re.split(r",", authors_raw.strip())[0]
    first_surname = re.sub(r"[^A-Za-z]", "", first_surname)
    return f"{first_surname}{year}"


def split_authors(authors_raw):
    authors_raw = authors_raw.strip().rstrip(",")
    # split off the final author joined with " and "
    m = re.search(r"^(.*?)\s+and\s+([^,]+,\s*[^,]+)$", authors_raw)
    if m:
        rest, last = m.group(1), m.group(2)
    else:
        rest, last = "", authors_raw
    units = []
    if rest:
        units = re.split(r"(?<=\.),\s*", rest.strip().rstrip(","))
    units = [u.strip() for u in units if u.strip()]
    units.append(last.strip())
    return " and ".join(units)


ENTRY_RE = re.compile(
    r"^(?P<authors>.+?)\s*\((?P<year>\d{4}[a-z]?)\),\s*(?P<rest>.+)$"
)
DOI_RE = re.compile(r"doi:\s*([^\s,]+)\.?\s*$")
VOL_RE = re.compile(r"Vol\.\s*(\d+)\s*No\.\s*([\w\-]+)")
PP_RE = re.compile(r"pp\.\s*([\d,]+)\s*(?:[-–—]+\s*([\d,]+))?")
QUOTED_RE = re.compile(r"[“\"]([^”\"]+)[”\"]")


def parse_entry(raw, key_seen):
    m = ENTRY_RE.match(raw)
    if not m:
        return None
    authors_raw, year, rest = m.group("authors"), m.group("year"), m.group("rest")
    year_num = re.sub(r"[a-z]$", "", year)

    doi = None
    dm = DOI_RE.search(rest)
    if dm:
        doi = dm.group(1).rstrip(".")
        rest = rest[: dm.start()].rstrip()

    title = None
    qm = QUOTED_RE.search(rest)
    if qm:
        title = qm.group(1)

    emph_m = re.search(r"@@(.+?)@@", rest)
    venue = emph_m.group(1) if emph_m else None

    vol_m = VOL_RE.search(rest)
    pp_m = PP_RE.search(rest)

    def pp_range():
        start = pp_m.group(1).rstrip(",")
        end = pp_m.group(2).rstrip(",") if pp_m.group(2) else None
        return f"{start}--{end}" if end else start

    plain_rest = re.sub(r"@@", "", rest)
    if title:
        plain_rest = plain_rest.replace(f"“{title}”", "").replace(f'"{title}"', "")
    plain_rest = re.sub(r"^[\s,]+", "", plain_rest).strip()

    authors_bib = split_authors(authors_raw)
    base_key = make_key(authors_raw, year)
    key = base_key
    n = 1
    while key in key_seen:
        n += 1
        key = f"{base_key}{chr(96+n)}"
    key_seen.add(key)

    fields = {"author": authors_bib, "year": year_num}
    if doi:
        fields["doi"] = doi

    if title and venue and vol_m:
        entry_type = "article"
        fields["title"] = title
        fields["journal"] = venue
        fields["volume"] = vol_m.group(1)
        fields["number"] = vol_m.group(2)
        if pp_m:
            fields["pages"] = pp_range()
    elif title and venue:
        entry_type = "article"
        fields["title"] = title
        fields["journal"] = venue
        if pp_m:
            fields["pages"] = pp_range()
    elif title and not venue:
        entry_type = "misc"
        fields["title"] = title
        fields["howpublished"] = plain_rest
    elif venue and not title:
        entry_type = "book"
        fields["title"] = venue
        after_venue = plain_rest.split(venue, 1)[-1].strip(" ,.")
        if after_venue:
            fields["publisher"] = after_venue
    else:
        return None

    return key, entry_type, fields


def format_bib(key, entry_type, fields):
    lines = [f"@{entry_type}{{{key},"]
    for k, v in fields.items():
        v = bibtex_escape(str(v))
        lines.append(f"  {k} = {{{v}}},")
    lines.append("}")
    return "\n".join(lines)


DUPLICATE_MERGES = {
    # (authors-prefix, year) pairs to drop, keeping the first occurrence
    "atwater2002b": "atwater2002a",
}


def main():
    raw_entries = get_reference_paragraphs()
    print(f"{len(raw_entries)} raw reference paragraphs found")

    key_seen = set()
    parsed = []
    warnings = []
    for raw in raw_entries:
        r = parse_entry(raw, key_seen)
        if r is None:
            warnings.append(raw)
            continue
        parsed.append((raw, *r))

    # Merge Atwater 2002a/2002b (identical DOI/journal/vol/pages -> true dup)
    atwater = [p for p in parsed if p[1].startswith("Atwater")]
    if len(atwater) == 2 and atwater[0][3].get("doi") == atwater[1][3].get("doi"):
        drop_key = atwater[1][1]
        parsed = [p for p in parsed if p[1] != drop_key]
        print(f"merged duplicate: dropped {drop_key} (identical DOI to {atwater[0][1]})")

    # Merge the two Cox/Schleier 2010 "Theory of Constraints Handbook" entries
    cox = [p for p in parsed if "Cox" in p[3]["author"] and "Constraints Handbook" in p[3].get("title", "")]
    if len(cox) == 2:
        keep, drop = cox[0], cox[1]
        parsed = [p for p in parsed if p[1] != drop[1]]
        # clean the kept entry: normalize author names, drop the bogus DOI if present
        _, key, etype, fields = keep
        fields["author"] = "Cox, J.F. and Schleier, J.G."
        fields.pop("doi", None)
        fields["publisher"] = "McGraw-Hill"
        fields["address"] = "New York"
        print(f"merged duplicate: dropped {drop[1]}, cleaned {key} (removed mismatched DOI)")

    # Generic dedup: entries sharing the same author string and base year
    # (i.e. only differing by a Mendeley a/b/c disambiguation suffix) whose
    # titles are identical verbatim are the same work counted twice -- keep
    # the first, drop the rest. (Same author+year with DIFFERENT titles,
    # e.g. Schragenheim's several 2010 book chapters, are left alone.)
    groups = {}
    for entry in parsed:
        raw, key, etype, fields = entry
        base_year = re.sub(r"[a-z]$", "", fields.get("year", ""))
        group_key = (fields.get("author", ""), base_year, fields.get("title", "").lower())
        groups.setdefault(group_key, []).append(entry)
    deduped = []
    for group_key, entries in groups.items():
        if len(entries) > 1:
            print(f"merged duplicate: kept {entries[0][1]}, dropped "
                  f"{', '.join(e[1] for e in entries[1:])} (identical title/author/year)")
        deduped.append(entries[0])
    parsed = deduped

    with open("references.bib", "w") as f:
        for _, key, etype, fields in parsed:
            f.write(format_bib(key, etype, fields))
            f.write("\n\n")
        for i, raw in enumerate(warnings):
            key = f"UNPARSED{i+1}"
            f.write(format_bib(key, "misc", {"note": raw, "title": "UNPARSED -- needs manual entry"}))
            f.write("\n\n")

    print(f"wrote references.bib: {len(parsed)} parsed entries, {len(warnings)} unparsed (flagged)")
    if warnings:
        print("\n--- UNPARSED ENTRIES (need manual fix in references.bib) ---")
        for w in warnings:
            print(" -", w[:150])


if __name__ == "__main__":
    main()
