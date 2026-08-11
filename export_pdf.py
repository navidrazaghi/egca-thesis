"""Update the document's fields in Word and export a PDF.

`build_thesis.py` writes the body, but the table of contents, the list of
figures, the list of tables and every caption number are Word fields: they carry
the text of whatever they held when the template was last saved until Word
recalculates them.  Only Word can do that, so this step drives it over COM
rather than reimplementing field evaluation.

Two things learned the hard way and encoded here:

  * The export fails with a read-only COMException if the target PDF is open in
    a viewer, so the output name carries a suffix and existing files are never
    silently overwritten while locked.
  * Fields have to be updated twice.  The first pass fills the entries in, which
    changes how many pages the front matter occupies, which changes the page
    numbers the entries point at; the second pass settles them.

Usage:
    python export_pdf.py [--docx Razaghi_MSc_Thesis.docx] [--suffix v3]
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", default="Razaghi_MSc_Thesis.docx")
    ap.add_argument("--suffix", default="v3")
    a = ap.parse_args()

    import win32com.client as win32
    from pywintypes import com_error

    src = os.path.abspath(a.docx)
    if not os.path.exists(src):
        sys.exit(f"not found: {src}")
    stem = os.path.splitext(src)[0]
    out_docx = f"{stem}_{a.suffix}.docx"
    out_pdf = f"{stem}_{a.suffix}.pdf"
    for path in (out_docx, out_pdf):
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                sys.exit(f"{path} is open in another program; close it first")

    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    doc = None
    try:
        doc = word.Documents.Open(src)
        for _ in range(2):
            doc.Fields.Update()
            for toc in doc.TablesOfContents:
                toc.Update()
            for tof in doc.TablesOfFigures:
                tof.Update()
        doc.SaveAs2(out_docx, FileFormat=16)          # wdFormatDocumentDefault
        doc.ExportAsFixedFormat(out_pdf, 17)          # wdExportFormatPDF
        pages = doc.ComputeStatistics(2)              # wdStatisticPages
        print(f"pages: {pages}")
    except com_error as e:
        sys.exit(f"Word refused the operation: {e}")
    finally:
        if doc is not None:
            doc.Close(SaveChanges=0)
        word.Quit()

    print(f"docx: {out_docx}")
    print(f"pdf:  {out_pdf}")


if __name__ == "__main__":
    main()
