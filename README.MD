# Overleaf project: RL-TOC manuscript

This is the LaTeX/Overleaf project generated from the manuscript in the
sibling repo `rl-article` (`RL_TOC - Manuscript - Rev01 - answ -
mendeley.docx` -- the canonical source; a companion `.pages` file and
several `.backup-*.docx` copies in that repo were not used).

## Layout

- `main.tex` -- entry point. `article` class, `biblatex` (author-year style,
  bibtex backend), no author/affiliation block (manuscript stays blinded).
- `sections/*.tex` -- one file per section/subsection, `\input` from
  `main.tex` in order.
- `figures/` -- the 11 embedded images, renamed `fig01.png`…`fig10.png`
  (Figures 6 and 7's sub-panels share files: `fig06a.png` + `fig06bcde.png`,
  `fig07abcd.png` -- see **Known limitations** below).
- `references.bib` -- 47 deduplicated entries, parsed from the docx's
  Mendeley bibliography.
- `markdown/` -- the durable Markdown intermediate:
  - `manuscript.md` -- raw, unedited pandoc docx→markdown output.
  - `manuscript.cleaned.md` -- after the Phase 3 structural cleanup (see
    below). This is a good place to keep revising prose before touching
    LaTeX -- rerun `split_sections.py` afterwards to regenerate `sections/`.
  - `manuscript.json` / `manuscript.cleaned.json` -- the pandoc AST behind
    the two `.md` files; the scripts operate on these, not the `.md` text.
- `scripts/` -- the conversion pipeline (see below).

## Rebuilding from the docx

Run this from inside this repo, pointing at the docx in the sibling
`rl-article` checkout (adjust the path if yours lives elsewhere):

```bash
pandoc "../rl-article/RL_TOC - Manuscript - Rev01 - answ - mendeley.docx" \
  -f docx -t json -o markdown/manuscript.json
python3 scripts/build_manuscript.py      # -> markdown/manuscript.cleaned.json
pandoc markdown/manuscript.cleaned.json -f json -t markdown --wrap=none \
  -o markdown/manuscript.cleaned.md
python3 scripts/refs2bib.py              # -> references.bib
python3 scripts/split_sections.py        # -> sections/*.tex
tectonic --outdir build main.tex         # -> build/main.pdf
```

`build_manuscript.py` does the structural work: converts the 18
equation-numbering layout tables into real `\tag`-ged `equation`
environments, converts the 2 figure sub-panel layout tables into
`subcaption`-based subfigure groups, keeps only the 3 real data tables,
fixes a numbering collision (two "Table 2." captions -> "Table 2." /
"Table 3."), and flags the duplicated Section 4.3 (a "[REVISED DRAFT]" and
an "ORIGINAL" version both still present in the source) instead of
resolving it.

If you edit the manuscript in Word again, re-run the whole chain above
against the new docx. If you just want to keep revising prose, edit
`markdown/manuscript.cleaned.md` directly and re-run `split_sections.py`
(it reads `manuscript.cleaned.json`, so hand-edits to the `.md` need porting
back, or edit `sections/*.tex` directly at that point -- the Markdown stage
is a staging area, not a live source once you're past it).

## Known limitations (flagged, not fixed)

- **Section 4.3 is duplicated on purpose.** A "REVISED DRAFT -- pending
  author review" version and the original version both appear, each
  highlighted. They describe different reward-function formulations (22 vs.
  50 state variables) and have not been reconciled -- the source itself
  contains a reviewer note flagging that Figure 6's caption doesn't match
  the revised reward equations. Decide which to keep (or merge) and delete
  the other before submission.
- **43 in-text citation placeholders** (rendered as highlighted "Reference"
  boxes) are unresolved -- the author's own TODOs, carried through
  untouched. Word's yellow-highlight formatting survives as
  `\colorbox{yellow!40}{...}` throughout, so anything still marked yellow in
  the PDF needs attention.
- **In-text citations are plain text, not `\citep{}`.** The 51 working
  Mendeley citations render correctly as "(Author, Year)" but are not
  linked to `references.bib` entries via LaTeX commands (`\nocite{*}` forces
  the full bibliography to print regardless). Wiring them up would need
  matching ~50 citation strings against `.bib` keys -- a reasonable
  follow-up, not done here.
- **Figures 6 and 7's sub-panels reuse the same image file** for most
  panels (`fig06bcde.png` for Figure 6(b)-(e), `fig07abcd.png` for all of
  Figure 7(a)-(d)) -- this is faithful to the source docx, which has the
  same placeholder image repeated across those panels rather than distinct
  charts. These need to be replaced with the real per-panel charts.
- **A handful of hyperref cross-references are dangling** (shown as red
  boxes in the PDF, e.g. some `Figure N`/`Table N` links) where the
  original Word bookmark anchor sat inside content this pipeline restructured.
  Cosmetic only -- doesn't block compiling.
- **A few OCR/extraction artifacts remain**, e.g. "st ochastic" (stray
  space) in the abstract -- a Word run-splitting artifact, not something
  this pipeline should silently "fix." Worth a proofreading pass in
  `markdown/manuscript.cleaned.md`.
- Some `Overfull \hbox` warnings (long unbreakable inline math/table rows) --
  cosmetic spacing only, harmless.
- Built with `tectonic` locally for verification, using the `bibtex` backend
  for biblatex (tectonic's bundled biblatex version didn't match the
  Homebrew `biber` version available here). Overleaf's own environment pins
  matching biblatex/biber versions, so switching back to
  `backend=biber` there should also work if preferred.
