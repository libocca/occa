(require 'cc-mode)

(eval-when-compile
  (require 'cc-langs)
  (require 'cc-fonts))

(eval-and-compile
  (c-add-language 'okl-mode 'c++-mode))

(c-lang-defconst c-primitive-type-kwds
  "Primitives"
  okl (append
       (append (c-lang-const c-primitive-type-kwds) nil)
       '("bool2"   "bool3"    "bool4"    "bool8"    "bool16"
         "char2"   "char3"    "char4"    "char8"    "char16"
         "short2"  "short3"   "short4"   "short8"   "short16"
         "int2"    "int3"     "int4"     "int8"     "int16"
         "long2"   "long3"    "long4"    "long8"    "long16"
         "float2"  "float3"   "float4"   "float8"   "float16"
         "double2" "double3"  "double4"  "double8"  "double16")))

(c-lang-defconst c-modifier-kwds
  okl (append
       (append (c-lang-const c-modifier-kwds) nil)
       '("kernel" "occaKernel"
         "occaFunction"
         "shared" "exclusive")))

(c-lang-defconst c-other-op-syntax-tokens
  "New tokens"
  okl (append
       (append (c-lang-const c-other-op-syntax-tokens) nil)
       '("#" "##"
         "::" "..." "@")))

(c-lang-defconst c-other-op-syntax-tokens
  okl (append '("@") (c-lang-const c-other-op-syntax-tokens)))

(c-lang-defconst c-block-comment-start-regexp
  okl "/[*+]")

;; Font keywords formats from C
(defconst okl-font-lock-keywords-1
  (c-lang-const c-matchers-1 okl)
  "Minimal highlighting for OKL mode.")

(defconst okl-font-lock-keywords-2
  (c-lang-const c-matchers-2 okl)
  "Fast normal highlighting for OKL mode.")

(defconst okl-font-lock-keywords-3
  (c-lang-const c-matchers-3 okl)
  "Accurate normal highlighting for OKL mode.")

(defvar okl-font-lock-keywords
  okl-font-lock-keywords-3
  "Default expressions to highlight in OKL mode.")

(defvar okl-mode-syntax-table
  nil
  "Syntax table used in okl-mode buffers.")

(or okl-mode-syntax-table
    (setq okl-mode-syntax-table
          (funcall (c-lang-const c-make-mode-syntax-table okl))))

(defvar okl-mode-abbrev-table
  nil
  "Abbreviation table used in okl-mode buffers.")

(c-define-abbrev-table 'okl-mode-abbrev-table
  '(("else" "else" c-electric-continued-statement 0)
    ("while" "while" c-electric-continued-statement 0)))

(defvar okl-mode-map
  (let ((map (c-make-inherited-keymap))) map)
  "Keymap used in okl-mode buffers.")

(easy-menu-define okl-menu okl-mode-map "OKL Mode Commands"
  (cons "OKL" (c-lang-const c-mode-menu okl)))

(add-to-list 'auto-mode-alist '("\\.okl\\'" . okl-mode))

(defun okl-mode ()
  "OKL Mode"
  (interactive)
  (kill-all-local-variables)
  (c-initialize-cc-mode t)
  (set-syntax-table okl-mode-syntax-table)
  (setq major-mode 'okl-mode
        mode-name "okl"
        local-abbrev-table okl-mode-abbrev-table
        abbrev-mode t)
  (use-local-map c-mode-map)
  (c-init-language-vars okl-mode)
  (c-common-init 'okl-mode)
  (easy-menu-add okl-menu)
  (run-hooks 'c-mode-common-hook)
  (run-hooks 'okl-mode-hook)
  (setq font-lock-keywords-case-fold-search t)
  (c-update-modeline))

(provide 'okl-mode)