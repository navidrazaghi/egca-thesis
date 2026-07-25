# -*- coding: utf-8 -*-
"""Assemble the complete thesis docx from the Sharif template."""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from docbuilder import Builder
import content_part1 as c1
import content_part2 as c2
import content_part3 as c3

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "Razaghi_MSc_Thesis.docx")

b = Builder()

c1.front_matter(b)      # sections 0-6
c1.lists_section(b)     # sections 7-10 (TOC, LOF, LOT, symbols)
c1.chapter1(b)          # section 11
c1.chapter2(b)          # section 12
c2.chapter3(b)          # section 13
c2.chapter4(b)          # section 14
c3.chapter5(b)          # section 15
c3.chapter6(b)          # section 16
c3.references(b)        # section 17
c3.glossary(b)          # section 18
c3.appendices(b)        # section 19
c3.english_pages(b)     # sections 20-21

HEADER_REPL = {
    "header11.xml": [("فصل دوم: مشخصات یک نوشته خوب", "فصل دوم: پیشینه پژوهش")],
    "header12.xml": [("فصل سوم: نگارش صحيح", "فصل سوم: مبانی نظری و فرمول‌بندی مسئله")],
    "header13.xml": [("ماشین‌نویسی صحیح ", "روش پیشنهادی"),
                     ("ماشین‌نویسی صحیح", "روش پیشنهادی")],
    "header14.xml": [("فصل پنجم: کنترل کیفیت گزارش", "فصل پنجم: نتایج شبیه‌سازی و ارزیابی")],
    "header15.xml": [("فصل ششم: نتيجه‌گيري ", "فصل ششم: جمع‌بندی و نتیجه‌گیری"),
                     ("فصل ششم: نتيجه‌گيري", "فصل ششم: جمع‌بندی و نتیجه‌گیری")],
}

FOOTER_REPL = {
    "footer3.xml": [
        ("نام و نام خانوادگي نويسنده(گان)، «عنوان پایان‌نامه یا پروژه درسی»، "
         "پایان‌نامه کارشناسی یا کارشناسی‌ارشد یا رساله دکتری یا پروژه درس ...،",
         "نوید رزاقی، «بهبود عملکرد و استواری سامانه‌های رانندگی خودران "
         "انتها-به-انتها با استفاده از یادگیری چندوجهی و ترکیب بهینه اطلاعات "
         "سنسورهای ناهمگون»، پایان‌نامه کارشناسی‌ارشد،"),
        (" ویرایش چهارم،", ""),
        (" استاد راهنما یا استاد درس: ...، دانشگاه صنعتي شريف، دانشكده...، ماه و سال.",
         " استاد راهنما: دکتر بابک خلج، دانشگاه صنعتي شريف، دانشكده مهندسی برق، تیر ۱۴۰۵."),
    ],
}

out = b.save(OUT, header_repl=HEADER_REPL, footer_repl=FOOTER_REPL)
print("saved:", out)
print("images:", len(b.images), "| footnotes:", len(b.footnotes),
      "| body chunks:", len(b.body))
