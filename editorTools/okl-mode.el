;; Add:
;;
;;    (load-file (concat (getenv "OCCA_DIR") "/editorTools/okl-mode.el"))
;;
;; to your ~/.emacs file to use [okl-mode] for .okl files

(define-derived-mode okl-mode c++-mode
  "OKL"
  "[O]CCA [K]ernel [L]anguage mode."
)

(defun add-okl-keywords()
  "Adds OKL keywords"
  ;
  (font-lock-add-keywords nil
    '(("\\<\\(occaKernel\\)"   . 'font-lock-keyword-face)
      ("\\<\\(occaFunction\\)" . 'font-lock-keyword-face)
      ("\\<\\(shared\\)"       . 'font-lock-keyword-face)
      ("\\<\\(exclusive\\)"    . 'font-lock-keyword-face)
      ("\\<\\(outer0\\)"       . 'font-lock-keyword-face)
      ("\\<\\(outer1\\)"       . 'font-lock-keyword-face)
      ("\\<\\(outer2\\)"       . 'font-lock-keyword-face)
      ("\\<\\(inner0\\)"       . 'font-lock-keyword-face)
      ("\\<\\(inner1\\)"       . 'font-lock-keyword-face)
      ("\\<\\(inner2)\\)"      . 'font-lock-keyword-face)
      )
    )
)

(add-hook 'okl-mode-hook 'add-okl-keywords)

(add-to-list 'auto-mode-alist '("\\.okl\\'" . okl-mode))
