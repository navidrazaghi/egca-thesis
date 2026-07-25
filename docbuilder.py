# -*- coding: utf-8 -*-
"""OOXML builder that regenerates document.xml of the Sharif thesis template
while preserving its styles, sections, headers and footers."""
import os, re, shutil, struct, zipfile
import latex2mathml.converter as l2m
from lxml import etree

BASE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(BASE, "template_unpacked")
XSL_PATH = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
_transform = etree.XSLT(etree.parse(XSL_PATH))

RLM = "‏"
ZWNJ = "‌"


def esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def latex_to_omml(latex):
    mml = l2m.convert(latex)
    omml = _transform(etree.fromstring(mml))
    s = etree.tostring(omml.getroot()).decode()
    # strip the mml namespace decl (unused) but keep m:
    s = s.replace(' xmlns:mml="http://www.w3.org/1998/Math/MathML"', "")
    return s


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


LATIN_RE = re.compile(r"[A-Za-z\\$][A-Za-z0-9\-\._/\\&:\+\*=\^{}\$]*"
                      r"(?:[ ][A-Za-z0-9\-\._/\\&:\+\*=\^{}\$]+)*")


class Builder:
    def __init__(self):
        self.body = []            # XML chunks
        self.footnotes = []       # (id, text)
        self.next_fn_id = 14      # template footnotes end at 13
        self.images = []          # (rid, media_name, src_path)
        self.next_img = 1
        self.chapter = 0          # current chapter number (Heading1 numbered)
        self.fig_num = 0
        self.tbl_num = 0
        self.eq_num = 0
        self.docpr = 100
        doc = open(os.path.join(TPL, "word", "document.xml"), encoding="utf-8").read()
        self.doc_open = doc[: doc.index("<w:body>") + 8]
        # extract the 22 sectPr blocks in order
        self.sects = re.findall(r"<w:sectPr[^>]*>.*?</w:sectPr>", doc, re.S)
        self.sect_i = 0

    # ------------------------------------------------------------------ runs
    def fa_run(self, text, bold=False, sz=None):
        rpr = '<w:rFonts w:hint="cs"/>'
        if bold:
            rpr += "<w:b/><w:bCs/>"
        if sz:
            rpr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
        rpr += "<w:rtl/>"
        return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'

    def en_run(self, text, bold=False, italic=False, sz=None):
        rpr = ""
        if bold:
            rpr += "<w:b/><w:bCs/>"
        if italic:
            rpr += "<w:i/><w:iCs/>"
        if sz:
            rpr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
        rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
        return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'

    def footnote_ref(self, en_text):
        fid = self.next_fn_id
        self.next_fn_id += 1
        self.footnotes.append((fid, en_text))
        return ('<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/><w:rtl/></w:rPr>'
                f'<w:footnoteReference w:id="{fid}"/></w:r>')

    def rich_runs(self, text, bold_all=False):
        """Convert marked-up Persian text to runs.
        Markers: **bold**, [[shown text|footnote EN]], Latin auto-detected."""
        out = []
        # split by footnote markers and inline math first
        parts = re.split(r"(\[\[.*?\]\]|\{\$.*?\$\})", text)
        for part in parts:
            if not part:
                continue
            if part.startswith("{$") and part.endswith("$}"):
                out.append(self.MATH_INLINE(part[2:-2]))
                continue
            m = re.match(r"\[\[(.*?)\|(.*?)\]\]$", part, re.S)
            if m:
                shown, fn = m.group(1), m.group(2)
                out.append(self._script_runs(shown, bold_all))
                out.append(self.footnote_ref(fn))
                continue
            # bold segments
            for seg in re.split(r"(\*\*.*?\*\*)", part):
                if not seg:
                    continue
                if seg.startswith("**") and seg.endswith("**"):
                    out.append(self._script_runs(seg[2:-2], True))
                else:
                    out.append(self._script_runs(seg, bold_all))
        return "".join(out)

    def _script_runs(self, text, bold=False):
        out = []
        pos = 0
        for m in LATIN_RE.finditer(text):
            # ignore single-char matches that are part of Persian words? fine.
            if m.start() > pos:
                out.append(self.fa_run(text[pos:m.start()], bold))
            out.append(self.en_run(m.group(0), bold))
            pos = m.end()
        if pos < len(text):
            out.append(self.fa_run(text[pos:], bold))
        return "".join(out)

    # ------------------------------------------------------------ paragraphs
    def para(self, style, runs_xml, extra_ppr=""):
        ppr = ""
        if style or extra_ppr:
            ps = f'<w:pStyle w:val="{style}"/>' if style else ""
            ppr = f"<w:pPr>{ps}{extra_ppr}</w:pPr>"
        self.body.append(f"<w:p>{ppr}{runs_xml}</w:p>")

    def P(self, text, style=None, align=None):
        extra = f'<w:jc w:val="{align}"/>' if align else ""
        self.para(style, self.rich_runs(text), extra)

    def PSTYLE(self, style, text=""):
        self.para(style, self.rich_runs(text) if text else "")

    def EMPTY(self, style=None, n=1):
        for _ in range(n):
            self.para(style, "")

    def H1(self, text, numbered=True):
        if numbered:
            self.chapter += 1
            self.fig_num = 0
            self.tbl_num = 0
            self.eq_num = 0
            self.para("Heading1", self.rich_runs(text))
        else:
            # template cancels the multilevel numbering with a direct numId=0
            self.para("Heading1NoNumber", self.rich_runs(text),
                      '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>')

    def H2(self, text):
        self.para("Heading2", self.rich_runs(text))

    def H3(self, text):
        self.para("Heading3", self.rich_runs(text))

    def BULLETS(self, items, style="Bulet"):
        for it in items:
            self.para(style, self.rich_runs(it))

    def SECT(self):
        """Close current section using the template's next sectPr."""
        sp = self.sects[self.sect_i]
        self.sect_i += 1
        if self.sect_i < len(self.sects):
            self.body.append(f"<w:p><w:pPr>{sp}</w:pPr></w:p>")
        else:
            self.body.append(sp)  # final body-level sectPr

    # ---------------------------------------------------------------- fields
    def _field(self, instr, cached_runs):
        return ('<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
                f'<w:r><w:instrText xml:space="preserve"> {esc(instr)} </w:instrText></w:r>'
                '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
                + cached_runs +
                '<w:r><w:fldChar w:fldCharType="end"/></w:r>')

    def _num_field_pair(self, seq_name, num):
        """STYLEREF chapter + '.' + SEQ number, with cached values."""
        ch_cached = (f'<w:r><w:rPr><w:noProof/><w:rtl/></w:rPr>'
                     f'<w:t>{RLM}{self.chapter}</w:t></w:r>')
        n_cached = (f'<w:r><w:rPr><w:noProof/><w:rtl/></w:rPr>'
                    f'<w:t>{num}</w:t></w:r>')
        return (self._field("STYLEREF 1 \\s", ch_cached)
                + f'<w:r><w:rPr><w:rtl/></w:rPr><w:t>{RLM}.{RLM}</w:t></w:r>'
                + self._field(f"SEQ {seq_name} \\* ARABIC \\s 1", n_cached))

    # ---------------------------------------------------------------- images
    def FIG(self, png, caption, width_cm=None):
        """Insert centered figure (InPicture) + caption below (PicTitle)."""
        src = os.path.join(BASE, "figs", png)
        w_px, h_px = png_size(src)
        w_cm = w_px / 300.0 * 2.54
        h_cm = h_px / 300.0 * 2.54
        if width_cm is None:
            width_cm = min(w_cm, 14.5)
        scale = width_cm / w_cm
        cx = int(width_cm / 2.54 * 914400)
        cy = int(h_cm * scale / 2.54 * 914400)
        rid = f"rIdFig{self.next_img}"
        name = f"thfig{self.next_img}.png"
        self.next_img += 1
        self.images.append((rid, name, src))
        self.docpr += 1
        drawing = (
            f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{self.docpr}" name="Figure {self.docpr}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks '
            f'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
            f'</wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{self.docpr}" name="Figure {self.docpr}"/>'
            f'<pic:cNvPicPr><a:picLocks noChangeAspect="1" noChangeArrowheads="1"/></pic:cNvPicPr>'
            f'</pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:srcRect/>'
            f'<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr bwMode="auto"><a:xfrm><a:off x="0" y="0"/>'
            f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/>'
            f'<a:ln><a:noFill/></a:ln></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            f'</wp:inline></w:drawing></w:r>')
        self.para("InPicture", drawing)
        self.fig_num += 1
        cap = (self.fa_run("شكل "))
        cap += self._num_field_pair("Figure", self.fig_num)
        cap += self.fa_run("  ")
        cap += self.rich_runs(caption)
        self.para("PicTitle", cap)

    # ---------------------------------------------------------------- tables
    def TBL(self, caption, headers, rows, widths=None, small=False, latin_cols=None):
        """Caption above (TableTitle) + booktabs-style table (TableStyle).
        headers: list of header texts; rows: list of row lists.
        latin_cols: set of column indices treated as Latin/LTR content."""
        self.tbl_num += 1
        cap = self.fa_run("جدول ")
        cap += self._num_field_pair("Table", self.tbl_num)
        cap += self.fa_run("  ")
        cap += self.rich_runs(caption)
        self.para("TableTitle", cap)
        ncol = len(headers)
        if widths is None:
            widths = [int(8720 / ncol)] * ncol
        latin_cols = latin_cols or set()
        cellstyle = "InTableCSmall" if small else "InTableC"
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        t = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableStyle"/><w:bidiVisual/>'
             '<w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/>'
             '<w:tblLook w:val="0420" w:firstRow="1" w:lastRow="0" w:firstColumn="0"'
             ' w:lastColumn="0" w:noHBand="1" w:noVBand="1"/></w:tblPr>'
             f'<w:tblGrid>{grid}</w:tblGrid>']
        # header row
        t.append('<w:tr><w:trPr><w:cnfStyle w:val="100000000000"/><w:tblHeader/></w:trPr>')
        for j, h in enumerate(headers):
            t.append(self._cell(h, widths[j], "InTableCB", j in latin_cols))
        t.append("</w:tr>")
        for row in rows:
            t.append("<w:tr>")
            for j, c in enumerate(row):
                t.append(self._cell(str(c), widths[j], cellstyle, j in latin_cols))
            t.append("</w:tr>")
        t.append("</w:tbl>")
        self.body.append("".join(t))
        self.EMPTY()  # spacing paragraph after table

    def _cell(self, text, w, style, latin):
        runs = self.rich_runs(text)
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>'
                f'<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{runs}</w:p></w:tc>')

    # ------------------------------------------------------------- equations
    def EQ(self, latex, punct=""):
        """Numbered display equation, template style: (ch.n) TAB equation."""
        self.eq_num += 1
        runs = self.fa_run("(")
        runs += self._num_field_pair("Equation", self.eq_num)
        runs += self.fa_run(")")
        runs += "<w:r><w:tab/></w:r>"
        runs += latex_to_omml(latex)
        if punct:
            runs += self.en_run(punct)
        self.para("Equation", runs)
        return f"{self.chapter}.{self.eq_num}"

    def MATH_INLINE(self, latex):
        return latex_to_omml(latex)

    def SYMROW(self, latex, desc):
        """One row of the symbols list: symbol TAB description."""
        runs = latex_to_omml(latex)
        runs += "<w:r><w:tab/></w:r>"
        runs += self.rich_runs(desc)
        self.para("Math", runs)

    # ------------------------------------------------------------------- TOC
    def TOC_FIELD(self, instr, entries):
        """entries: list of (style, text_xml_runs, page). Static cached entries
        wrapped in a real TOC field so Word can update them (F9)."""
        first_style, first_runs, first_pg = entries[0]
        begin = ('<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
                 f'<w:r><w:instrText xml:space="preserve"> {esc(instr)} </w:instrText></w:r>'
                 '<w:r><w:fldChar w:fldCharType="separate"/></w:r>')
        tab = ('<w:r><w:rPr><w:noProof/><w:webHidden/></w:rPr><w:tab/></w:r>')
        def pg_runs(pg):
            return (f'<w:r><w:rPr><w:noProof/><w:webHidden/><w:rtl/></w:rPr>'
                    f'<w:t>{pg}</w:t></w:r>')
        n = len(entries)
        for i, (style, runs, pg) in enumerate(entries):
            content = runs + tab + pg_runs(pg)
            if i == 0:
                content = begin + content
            if i == n - 1:
                content += '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            self.para(style, content)

    def toc_entry_runs(self, num, title):
        r = ""
        if num:
            r += (f'<w:r><w:rPr><w:noProof/><w:rtl/></w:rPr>'
                  f'<w:t xml:space="preserve">{esc(num)}</w:t></w:r>'
                  '<w:r><w:rPr><w:noProof/></w:rPr><w:tab/></w:r>')
        r += self.rich_runs(title)
        return r

    # ------------------------------------------------------------------ save
    def save(self, out_path, header_repl=None, footer_repl=None):
        workdir = os.path.join(BASE, "build_unpacked")
        if os.path.exists(workdir):
            shutil.rmtree(workdir)
        shutil.copytree(TPL, workdir)
        assert self.sect_i == len(self.sects), \
            f"used {self.sect_i} of {len(self.sects)} sections"
        doc = self.doc_open + "".join(self.body) + "</w:body></w:document>"
        open(os.path.join(workdir, "word", "document.xml"), "w",
             encoding="utf-8").write(doc)
        # footnotes
        fpath = os.path.join(workdir, "word", "footnotes.xml")
        fx = open(fpath, encoding="utf-8").read()
        add = []
        for fid, text in self.footnotes:
            add.append(
                f'<w:footnote w:id="{fid}"><w:p><w:pPr>'
                f'<w:pStyle w:val="FootnoteText"/><w:bidi w:val="0"/></w:pPr>'
                f'<w:r><w:rPr><w:rStyle w:val="FootnoteReference"/></w:rPr>'
                f'<w:footnoteRef/></w:r>'
                f'<w:r><w:t xml:space="preserve"> {esc(text)}</w:t></w:r></w:p></w:footnote>')
        fx = fx.replace("</w:footnotes>", "".join(add) + "</w:footnotes>")
        open(fpath, "w", encoding="utf-8").write(fx)
        # images: media + rels
        for rid, name, src in self.images:
            shutil.copy(src, os.path.join(workdir, "word", "media", name))
        rpath = os.path.join(workdir, "word", "_rels", "document.xml.rels")
        rx = open(rpath, encoding="utf-8").read()
        addr = "".join(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
            f'officeDocument/2006/relationships/image" Target="media/{name}"/>'
            for rid, name, _ in self.images)
        rx = rx.replace("</Relationships>", addr + "</Relationships>")
        open(rpath, "w", encoding="utf-8").write(rx)
        # headers / footers replacements
        for fname, repls in (header_repl or {}).items():
            p = os.path.join(workdir, "word", fname)
            hx = open(p, encoding="utf-8").read()
            for old, new in repls:
                hx = hx.replace(esc(old), esc(new))
            open(p, "w", encoding="utf-8").write(hx)
        for fname, repls in (footer_repl or {}).items():
            p = os.path.join(workdir, "word", fname)
            hx = open(p, encoding="utf-8").read()
            for old, new in repls:
                hx = hx.replace(esc(old), esc(new))
            open(p, "w", encoding="utf-8").write(hx)
        # settings: auto-update fields on open
        spath = os.path.join(workdir, "word", "settings.xml")
        sx = open(spath, encoding="utf-8").read()
        if "<w:updateFields" not in sx and "<w:compat>" in sx:
            sx = sx.replace("<w:compat>", '<w:updateFields w:val="true"/><w:compat>')
            open(spath, "w", encoding="utf-8").write(sx)
        # zip
        if os.path.exists(out_path):
            os.remove(out_path)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(workdir):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, workdir).replace("\\", "/")
                    z.write(full, rel)
        return out_path
