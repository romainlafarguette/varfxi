(TeX-add-style-hook
 "slides"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("inputenc" "utf8") ("fontenc" "T1") ("caption" "font=scriptsize" "labelfont=scriptsize" "labelfont={color=imfblue}")))
   (TeX-run-style-hooks
    "latex2e"
    "../output/regressions_table_short"
    "beamer"
    "beamer10"
    "babel"
    "inputenc"
    "fontenc"
    "lmodern"
    "amsfonts"
    "amsmath"
    "mathabx"
    "bm"
    "bbm"
    "graphicx"
    "subfig"
    "xcolor"
    "booktabs"
    "rotating"
    "multirow"
    "pdflscape"
    "afterpage"
    "threeparttable"
    "caption"
    "import"
    "appendixnumberbeamer"
    "natbib")
   (TeX-add-symbols
    '("myframe" 1))
   (LaTeX-add-environments
    "largeitemize"
    "largeenumerate"
    "wideitemize"
    "wideenumerate"))
 :latex)

