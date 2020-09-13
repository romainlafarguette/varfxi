(TeX-add-style-hook
 "slides"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("babel" "english") ("inputenc" "utf8") ("fontenc" "T1") ("caption" "font=scriptsize" "labelfont=scriptsize" "labelfont={color=imfblue}")))
   (add-to-list 'LaTeX-verbatim-environments-local "semiverbatim")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
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
   (LaTeX-add-environments
    "largeitemize"
    "largeenumerate"
    "wideitemize"
    "wideenumerate")
   (LaTeX-add-xcolor-definecolors
    "imfblue"))
 :latex)

