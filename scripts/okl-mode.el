(require 'cc-mode)

(eval-when-compile
  (require 'cc-langs)
  (require 'cc-fonts))

(eval-and-compile
  (c-add-language 'okl-mode 'objc-mode))

(c-lang-defconst c-primitive-type-kwds
  "Primitives"
  okl (append
       (append (c-lang-const c-primitive-type-kwds) nil)
       '("bool2"   "bool3"   "bool4"
         "char2"   "char3"   "char4"
         "short2"  "short3"  "short4"
         "int2"    "int3"    "int4"
         "long2"   "long3"   "long4"
         "float2"  "float3"  "float4"
         "double2" "double3" "double4")))

;; Required defines for extending C++ mode
(c-lang-defconst c-opt-friend-key
  okl (c-lang-const c-opt-friend-key c++))

(c-lang-defconst c-opt-inexpr-brace-list-key
  okl (c-lang-const c-opt-inexpr-brace-list-key c++))

(c-lang-defconst c-opt-postfix-decl-spec-key
  okl (c-lang-const c-opt-postfix-decl-spec-key c++))

;; Syntax highlighting
(defcustom okl-font-lock-extra-types nil
  "*List of extra types (aside from the type keywords) to recognize in OKL mode.
Each list item should be a regexp matching a single identifier.")

(defconst okl-font-lock-keywords-1 (c-lang-const c-matchers-1 c++)
  "Minimal highlighting for OKL mode.")

(defconst okl-font-lock-keywords-2 (c-lang-const c-matchers-2 c++)
  "Fast normal highlighting for OKL mode.")

;; Set @annotation coloring
(defconst okl-font-lock-keywords-3
  (append
   (c-lang-const c-matchers-3 c++)
   `((eval . (list "\\<\\(@[a-zA-Z][a-zA-Z0-9_]*\\)\\>" 1 c-annotation-face))))
  "Accurate normal highlighting for OKL mode.")

(defvar okl-font-lock-keywords okl-font-lock-keywords-3
  "Default expressions to highlight in OKL mode.")

(defvar okl-mode-map
  (let ((map (c-make-inherited-keymap)))
    map)
  "Keymap used in okl-mode buffers.")

(easy-menu-define okl-menu okl-mode-map "OKL Mode Commands"
  (cons "OKL" (c-lang-const c-mode-menu okl)))

(defvar okl-mode-syntax-table nil
  "Syntax table used in okl-mode buffers.")
(or okl-mode-syntax-table
    (setq okl-mode-syntax-table
          (funcall (c-lang-const c-make-mode-syntax-table c++))))

(defvar okl-mode-abbrev-table nil
  "Abbreviation table used in okl-mode buffers.")
(c-define-abbrev-table 'okl-mode-abbrev-table
  '())

(defun okl-mode ()
  "Major mode for editing OKL"
  (interactive)
  (kill-all-local-variables)
  (c-initialize-cc-mode t)
  (set-syntax-table okl-mode-syntax-table)
  (setq major-mode 'okl-mode
        mode-name "OKL"
        local-abbrev-table okl-mode-abbrev-table
        abbrev-mode t)
  (use-local-map c++-mode-map)
  (c-init-language-vars okl-mode)
  (c-common-init 'okl-mode)
  (easy-menu-add okl-menu)
  (run-hooks 'c-mode-common-hook)
  (run-hooks 'okl-mode-hook)
  (c-update-modeline))

(setq okl-mode-hook ())
(add-hook 'okl-mode-hook
          (lambda ()
            (c-set-offset 'annotation-var-cont 0)
            ; Set Java @annotation indentation (hard-coded in cc-engine :()
            (setq c-buffer-is-cc-mode 'java-mode)
            ))

(add-to-list 'auto-mode-alist '("\\.okl\\'" . okl-mode))

(provide 'derived-mode-ex)
