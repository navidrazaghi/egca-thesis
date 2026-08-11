# -*- coding: utf-8 -*-
"""Confirm the exported PDF actually rendered its maths and figures.

The build pipeline turns each equation into an image. When that step fails it
does not raise: the LaTeX source lands in the page as literal text and the file
opens perfectly well, so the failure is invisible unless someone reads every
formula. This happened once already in this project, with a whole document of
half-processed markup that looked fine at a glance.

Usage:
    python check_pdf_render.py Razaghi_MSc_Thesis_v6.pdf
"""
import re
import sys

import fitz

RAW = re.compile(r"\\(frac|sum|mathrm|left|right|sigma|phi|mathbb|times|cdot|top)")
DOLLAR = re.compile(r"\$[^$\n]{2,}\$")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "Razaghi_MSc_Thesis_v6.pdf"
    d = fitz.open(path)
    pages = [d[i].get_text() for i in range(d.page_count)]

    raw = [i + 1 for i, t in enumerate(pages) if RAW.search(t)]
    dol = [i + 1 for i, t in enumerate(pages) if DOLLAR.search(t)]
    imgs = [(i + 1, len(d[i].get_images())) for i in range(d.page_count)
            if d[i].get_images()]

    print("pages: %d" % d.page_count)
    print("  raw LaTeX commands visible : %s" % (raw or "none"))
    print("  $...$ markup visible       : %s" % (dol or "none"))
    print("  pages carrying images      : %d (%d images total)"
          % (len(imgs), sum(n for _, n in imgs)))

    # every figure the text promises should exist as a caption
    # both spellings of kaf occur in Persian text and the two are
    # different code points; counting only one silently reports zero
    caps = sum(t.count("شكل") + t.count("شکل")
               for t in pages)
    tbls = sum(t.count("جدول") for t in pages)
    print("  figure captions + list rows: %d" % caps)
    print("  table captions + list rows : %d" % tbls)

    ok = not raw and not dol
    print("\n%s" % ("maths and figures rendered" if ok
                    else "RENDER FAILURE: formulas left as source text"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
