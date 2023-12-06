#lang racket

(require racket/hash)
(require racket/cmdline)

(require "souffle_ir.rkt")
; (require "gdlog_ir.rkt")
; (define (souffle-ram->gdlog-ir ram)
;     )


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; NOTE: this is one-off cases for ddisasm
; type is not supported
(define (extra-type-convert type-s)
  (cond
    [(string-prefix? type-s "r:stack_var") "STRING_NUM"]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; pass q.0 simplify IR, flatten nested for loop/and etc.

(define (ungrounded-vars-index args)
  (for/fold ([ungounded-vars '()])
            ([id (in-range (length args))]
             [arg (in-list args)])
    (match arg
      ['UNDEF  (cons id ungounded-vars)]
      [_ ungounded-vars])))

;; populate all RA operations from souffle RAM IR's subroutines

; -> list of q-mir rules
(define (populate-ra-op-stmt-loop stmt)
  (match stmt
    [`(STMTS ,(? ram-statement? seq-stmts) ...) (populate-ra-op-stmts-loop seq-stmts)]
    ; [`(QUERY ,(? ram-operation? ram-op))
    ;  (let-values ([(ops var-bindings) (populate-ra-op ram-op)])
    ;    `(QUERY ,ops ,var-bindings))]
    [`(DEBUG ,souffle-code (QUERY ,(? ram-operation? ram-op)))
     (let-values ([(ops var-bindings) (populate-ra-op ram-op)])
       `((RECURSIVE ,souffle-code ,ops ,var-bindings)))]
    [`(TIMER_ON ,(? symbol? rel-name) ,(? string? msg) ,(? ram-statement? stmt))
     (populate-ra-op-stmt-loop stmt)]
    [_ '()]))

; -> list of q-mir rules
(define (populate-ra-op-stmt stmt)
  (match stmt
    [`(LOOP ,(? ram-statement? stmt))
     `((FIXPOINT-LOOP ,(populate-ra-op-stmt-loop stmt)))]
    [`(DEBUG ,souffle-code (QUERY ,(? ram-operation? ram-op)))
     (let-values ([(ops var-bindings) (populate-ra-op ram-op)])
       `((ONCE-RULE ,souffle-code ,ops ,var-bindings)))]
    [`(DEBUG ,souffle-code (TIMER_ON ,_ ,_ (QUERY ,(? ram-operation? ram-op))))
     (let-values ([(ops var-bindings) (populate-ra-op ram-op)])
       `((ONCE-RULE ,souffle-code ,ops ,var-bindings)))]
    ; [`(TIMER_ON ,(? symbol? rel-name) ,(? string? msg) ,(? ram-operation? op))
    ;  (populate-ra-op-stmt stmt)]
    [`(TIMER_ON ,(? symbol? rel-name) ,(? string? msg) ,(? ram-statement? stmt))
     (populate-ra-op-stmt stmt)]
    ; [`(QUERY ,(? ram-operation? ram-op))
    ;  (let-values ([(ops var-bindings) (populate-ra-op ram-op)])
    ;    `(QUERY ,ops ,var-bindings))]
    [`(CLEAR ,(? symbol? rel-name))
     (match (ram-relation-name->q-mir-relation-ref rel-name)
       [`(REL ,(? symbol? name) FULL)
        `((GC ,(ram-relation-name->q-mir-relation-ref rel-name)))]
       [`(REL ,(? symbol? name) DELTA)
        `((MODIFIED ,name))]
       [`(REL ,(? symbol? name) NEW)
        `()])]
    [`(LOGSIZE ,rel-name ,debug-msg) '()]
    [`(IO ,rel-name ,args) (list (simplify-io stmt))]
    [`(STMTS ,(? ram-statement? seq-stmts) ...) (populate-ra-op-stmts seq-stmts)]
    [_ 
     (displayln (format "nothing populated for stmt: ~a" stmt))
     '()]))

; remove redundant info in IO
(define (simplify-io io)
  (match io
    [`(IO ,(? symbol? rel-name) ,(? list? args))
     `(IO ,rel-name
          ,(filter (lambda (arg)
                     (match arg
                       [`(= IO ,type-s) #t]
                       [`(= filename ,file-s) #t]
                       [`(= deliminator ,de-s) #t]
                       [`(= fact-dir ,dir-s) #t]
                       [`(= operation ,op) #t]
                       [_ #f]))
                   args))]))

; compute list of query in cond and var binding (hash of tuple-id -> rel-name)
(define (populate-ra-op-cond r-cond)
  (match r-cond
    [`(NOT (EXISTS ,rel-name ,vars))
     (let ([ungrounded-ids (ungrounded-vars-index vars)])
       (values `((NEGATE ,(ram-relation-name->q-mir-relation-ref rel-name)
                         ,ungrounded-ids ,vars))
               (hash)))]
    [`(CONSTRAINT ,(? symbol? bin-op)
                  ,(? ram-expression? lhs) ,(? ram-expression? rhs))
     (values (list r-cond) (hash))]
    [`(AND ,(? ram-condition? lhs) ,(? ram-condition? rhs))
     (let-values ([(lhs-op lhs-env) (populate-ra-op-cond lhs)]
                  [(rhs-op rhs-env) (populate-ra-op-cond rhs)])
       (values (append lhs-op rhs-op)
               (hash-union lhs-env rhs-env)))]
    [_ (displayln (format "nothing populated for cond: ~a" r-cond))
       (values '() (hash))]))


; TODO: index relation name and aggregation relation name has issue
(define (populate-ra-op op)
  (match op
    [`(IF ,(? ram-condition? r-cond) ,(? ram-nested-operation? op))
     (let-values ([(op-cond cond-env) (populate-ra-op-cond r-cond)]
                  [(op-body body-env) (populate-ra-op op)])
       (values (append op-cond op-body) (hash-union cond-env body-env)))]
    [`(INDEXED_IF_EXISTS ,(? symbol? tuple-id) ,(? symbol? rel-name)
                         ,(? ram-index-operation? index) ,(? ram-condition? r-cond)
                         ,(? ram-nested-operation? op))
     ]
    [`(FOR ,(? symbol? tuple-id) ,(? symbol? rel-name) ,(? ram-index-operation? idx-op)
           ,(? ram-nested-operation? nested-op))
     (define-values (body-ops body-env) (populate-ra-op nested-op))
     (match idx-op
       [`(INDEX ,(? ram-indices-comp? exprs) ...)
        (values (cons `(SCAN ,(ram-relation-name->q-mir-relation-ref rel-name) ,tuple-id ,exprs)
                      body-ops)
                (hash-set body-env tuple-id
                          (ram-relation-name->q-mir-relation-ref rel-name)))]
       [`(INDEX_AGGREGATE (= (T ,tuple-id 0) ,(? ram-abstract-aggregation? agg))
                          ,(? symbol? tuple-id) ,(? ram-condition? condition)
                          ,(? ram-index-operation? index-op))
        (define aggregate-op
          `(AGGREGATION ,tuple-id ,agg ,condition ,index-op))
        (values (cons aggregate-op body-ops)
                (hash-set body-env tuple-id aggregate-op))])]
    [`(FOR_IN ,(? symbol? tuple-id) ,(? symbol? rel-name) ,(? ram-nested-operation? op))
     (let-values ([(op-body op-env) (populate-ra-op op)]
                  [(scan-op) `(FULL-SCAN ,(ram-relation-name->q-mir-relation-ref rel-name)
                                         ,tuple-id)])
       (values (cons scan-op op-body)
               (hash-set op-env tuple-id
                         (ram-relation-name->q-mir-relation-ref rel-name))))]
    [`(INSERT (,(? ram-expression? exprs)  ...) ,(? symbol? rel-name))
     (values `((GENERATE ,(ram-relation-name->q-mir-relation-ref rel-name) ,exprs))
             (hash))]))

(define (populate-ra-op-stmts stmts)
  (for/fold ([populated-ops '()])
            ([stmt (in-list stmts)])
    (append populated-ops (populate-ra-op-stmt stmt))))

(define (populate-ra-op-stmts-loop stmts)
  (for/fold ([populated-ops '()])
            ([stmt (in-list stmts)])
    (append populated-ops (populate-ra-op-stmt-loop stmt))))


(define (ram-subroutine->q-mir-scc subroutine)
  (match subroutine
    [`(SUBROUTINE ,(? symbol? name) ,(? ram-statement? stmt))
     (define scc-name-s (string-append "stratum_" (symbol->string name)))
     `(SCC ,(string->symbol scc-name-s) ,(populate-ra-op-stmt stmt))]
    [`(SUBROUTINE ,(? symbol? name))
     `(SCC ,(string->symbol (string-append "stratum_" (symbol->string name))) ())]
    [_ (error (displayln (format "expect a subroutine but got ~a" subroutine)))]))


(define (ram-relation-name->q-mir-relation-ref name)
  (define name-s (symbol->string name))
  (cond
    [(string-prefix? name-s "@delta_") `(REL ,(string->symbol (substring name-s 7)) DELTA)]
    [(string-prefix? name-s "@new_") `(REL ,(string->symbol (substring name-s 5)) NEW)]
    [else `(REL ,name FULL)]))

; remove complex type only leave raw type
(define (simplify-attr attr)
  (match attr
    [`(ATTRIBUTE ,name ,type)
     (define type-s (symbol->string type))
     (cond
       [(string-prefix? type-s "i:") `(ATTRIBUTE ,name INT)]
       [(string-prefix? type-s "u:") `(ATTRIBUTE ,name UNSIGNED)]
       [(string-prefix? type-s "s:") `(ATTRIBUTE ,name STRING)]
       [else `(ATTRIBUTE ,name ,(extra-type-convert type-s))])]
    [`(AUXILIARY ,name ,type)
     (define type-s (symbol->string type))
     (cond
       [(string-prefix? type-s "i:") `(ATTRIBUTE ,name INT)]
       [(string-prefix? type-s "u:") `(ATTRIBUTE ,name UNSIGNED)]
       [(string-prefix? type-s "s:") `(ATTRIBUTE ,name STRING)]
       [else `(ATTRIBUTE ,name ,(extra-type-convert type-s))])]))

(define (ram-relations->q-mir relations)
  ; (define rel-names (all-relation-names relations))
  (for/fold ([q-ir-rels '()])
            ([souffle-rel-decl (in-list relations)])
    (match souffle-rel-decl
      [`(RELATION ,name (,(? ram-relation-attr? attrs) ...) ,_ ...)
       (cond
         [(string-prefix? (symbol->string name) "@delta_" ) q-ir-rels]
         [(string-prefix? (symbol->string name) "@new_") q-ir-rels]
         [(string-prefix? (symbol->string name) "@delete_") q-ir-rels]
         [(string-prefix? (symbol->string name) "@reject_") q-ir-rels]
         [else (append q-ir-rels `((RELATION ,name ,(map simplify-attr attrs))))])])))

(define (ram-call->q-mir-exec call)
  (match call
    [`(CALL ,(? symbol? name))
     `(EXECUTE ,name)]))
(define (ram-calls->q-mir-exec calls)
  (match calls
    [`(STMTS (TIMER ,_ (STMTS ,calls ...))) (map ram-call->q-mir-exec calls)]
    [`(STMTS ,calls ...) (map ram-call->q-mir-exec calls)]))

(define (ram-program->q-mir ram)
  ; (displayln (format "length of ram: ~a" (length ram)))
  (match ram
    [`(PROGRAM (DECLARATION ,(? ram-relation? relations) ...)
               ( ,(? ram-subroutine? subroutines) ...)
               (MAIN ,calls))
     `(PROGRAM ,(ram-relations->q-mir relations)
               ,(map ram-subroutine->q-mir-scc subroutines)
               ,(ram-calls->q-mir-exec calls))]))


(define (tuples-in-args args)
  (for/fold ([tuples '()])
            ([arg (in-list args)])
    (match arg
      [`(TUPLE ,(? symbol? tuple-id) ,_)
       (if (member tuple-id tuples)
           tuples
           (append tuples (list tuple-id)))]
      [_ tuples])))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; pass q.1 remove existence check of tuple while insert into newt
; q-mir -> q2-mir
(define (remove-newt-exist-check-seq seq-rules)
  (match seq-rules
    ['() '()]
    [`(,(? q-mir-negate? negs) ...
       (GENERATE (REL ,res-rel-name NEW) (,(? ram-expression? res-exprs) ...)))
     (define new-negs
       (for/fold ([new-negs '()])
                 ([neg (in-list negs)])
         (match neg
           [`(NEGATE (REL ,neg-rel-name FULL) ,(? list? ungrounded-ids) (,(? ram-expression? neg-vars) ...))
            (if (and (equal? res-rel-name neg-rel-name)
                     (equal? neg-vars res-exprs))
                new-negs
                (append new-negs (list neg)))]
           [_ new-negs])))
     (append new-negs (list `(GENERATE (REL ,res-rel-name NEW) ,res-exprs)))]
    ; recursive case, these need trim is usually at the end
    [`(,hd ,tl ...) (cons hd (remove-newt-exist-check-seq tl))]))

(define (remove-newt-exist-expr expr)
  (match expr
    [`(ONCE-RULE ,src-code (,(? q-mir-operation? ops) ...) ,(? hash? var-bindings))
     `(ONCE-RULE ,src-code ,(remove-newt-exist-check-seq ops) ,var-bindings)]
    [`(RECURSIVE ,src-code (,(? q-mir-operation? ops) ...) ,(? hash? var-bindings))
     `(RECURSIVE ,src-code ,(remove-newt-exist-check-seq ops) ,var-bindings)]
    [`(FIXPOINT-LOOP (,(? q-mir-expr? rules) ...))
     `(FIXPOINT-LOOP ,(map remove-newt-exist-expr rules))]
    [_ expr]))

(define (q-mir->q-mir-2 mir)
  (match mir
    [`(PROGRAM (,(? q-mir-relation? relations) ...)
               (,(? q-mir-scc? sccs) ...)
               (,(? q-mir-exec? execs) ...))
     (define new-sccs
       (for/fold ([new-sccs '()])
                 ([scc (in-list sccs)])
         (match scc
           [`(SCC ,(? symbol? name) (,(? q-mir-expr? exprs) ...))
            (append new-sccs `((SCC ,name ,(map remove-newt-exist-expr exprs))))]
           [_ (displayln (format "expect a scc but got ~a" scc))])))
     `(PROGRAM ,relations
               ,new-sccs
               ,execs)]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; pass q.2 remove unnecessary existence check in self join
; q2-mir -> q3-mir
(define (q-mir-2->q-mir-3 mir)
  (match mir
    [`(PROGRAM (,(? q-mir-relation? relations) ...)
               (,(? q-mir-scc? sccs) ...)
               (,(? q-mir-exec? execs) ...))
     (define new-sccs
       (for/fold ([new-sccs '()])
                 ([scc (in-list sccs)])
         (match scc
           [`(SCC ,(? symbol? name) (,(? q-mir-expr? exprs) ...))
            (append new-sccs `((SCC ,name ,(map remove-self-join-exist-expr exprs))))]
           [_ (displayln (format "expect a scc but got ~a" scc))])))
     `(PROGRAM ,relations
               ,new-sccs
               ,execs)]))

(define (remove-self-join-check seq-rules env eq-list)
  (define (ref-have-same-name? r1 r2)
    (equal? (second r1) (second r2)))
  (match seq-rules
    ['() '()]
    [`((NEGATE (REL ,neg-rel-name ,neg-ver) ,(? list? ungrounded-ids)
               (,(? ram-expression? neg-vars) ...))
       ,rst ...)
     (define nref `(REL ,neg-rel-name ,neg-ver))
     (define need-drop?
       (for/fold ([drop-flag #t])
                 ([neg-var (in-list neg-vars)]
                  [idx (in-range (length neg-vars))])
         (match neg-var
           [`(TUPLE ,tp ,tp-i)
            (and drop-flag
                 (or (and (ref-have-same-name? (hash-ref env tp) nref)
                          (equal? tp-i idx))
                     (for/fold ([eq-flag #f])
                               ([tp-eq (in-list eq-list)])
                       (match tp-eq
                         [`(= ,t1 ,t2)
                          (cond
                            [(equal? t1 neg-var)
                             (match t2
                               [`(TUPLE ,t2-rel ,t2-i)
                                (define t2-rel-name (second (hash-ref env t2-rel)))
                                (or eq-flag
                                    (and (equal? t2-rel-name neg-rel-name)
                                         (equal? t2-i idx)))])]
                            [(equal? t2 neg-var)
                             (match t1
                               [`(TUPLE ,t1-rel ,t1-i)
                                (define t1-rel-name (second (hash-ref env t1-rel)))
                                (or eq-flag
                                    (and (equal? t1-rel-name neg-rel-name)
                                         (equal? t1-i idx)))])]
                            [else eq-flag])]
                         [_ eq-flag]))))]
           [_ #f])))
     (if need-drop?
         (remove-self-join-check rst env eq-list)
         (cons `(NEGATE (REL ,neg-rel-name FULL) ,ungrounded-ids ,neg-vars)
               (remove-self-join-check rst env eq-list)))]
    [`((SCAN ,rel-ref ,tp-name ,index-list) ,rst ...)
     (cons `(SCAN ,rel-ref ,tp-name ,index-list)
           (remove-self-join-check rst env (append eq-list index-list)))]
    [`(,hd ,tl ...)
     (cons hd (remove-self-join-check tl env eq-list))]))


(define (remove-self-join-exist-expr expr)
  (match expr
    [`(ONCE-RULE ,src-code (,(? q-mir-operation? ops) ...) ,(? hash? var-bindings))
     `(ONCE-RULE ,src-code ,(remove-self-join-check ops var-bindings '()) ,var-bindings)]
    [`(RECURSIVE ,src-code (,(? q-mir-operation? ops) ...) ,(? hash? var-bindings))
     `(RECURSIVE ,src-code ,(remove-self-join-check ops var-bindings '()) ,var-bindings)]
    [`(FIXPOINT-LOOP (,(? q-mir-expr? rules) ...))
     `(FIXPOINT-LOOP ,(map remove-self-join-exist-expr rules))]
    [_ expr]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; q-mir->ra-mir

; pass ra.0 convert all full-scan-gen into copy

; -> list of ra-mir rules
(define (eliminate-full-scan-gen-seq mir-seq)
  (match mir-seq
    [`((FULL-SCAN ,(? q-mir-relation-ref? rel-src) ,(? symbol? tuple-id))
       (GENERATE ,rel-dest (,(? ram-expression? exprs) ...)))
     (list `(COPY ,rel-src ,rel-dest ,exprs))]
    [_ mir-seq]))

(define (eliminate-full-scan-gen-expr mir-expr)
  (match mir-expr
    [`(ONCE-RULE ,src-code (,(? q-mir-operation? ops) ...) ,(? hash? var-bindings))
     `(ONCE-RULE ,src-code ,(eliminate-full-scan-gen-seq ops) ,var-bindings)]
    [`(RECURSIVE ,src-code (,(? q-mir-operation? ops) ...) ,(? hash? var-bindings))
     `(RECURSIVE ,src-code ,(eliminate-full-scan-gen-seq ops) ,var-bindings)]
    [`(FIXPOINT-LOOP (,(? q-mir-expr? rules) ...))
     `(FIXPOINT-LOOP ,(map eliminate-full-scan-gen-expr rules))]
    [_ mir-expr]))

(define (q-mir-3->ra-mir-0 mir)
  (match mir
    [`(PROGRAM (,(? q-mir-relation? relations) ...)
               (,(? q-mir-scc? sccs) ...)
               (,(? q-mir-exec? execs) ...))
     `(PROGRAM ,relations
               ,(for/fold ([new-sccs '()])
                          ([scc (in-list sccs)])
                  (match scc
                    [`(SCC ,(? symbol? name) (,(? q-mir-expr? exprs) ...))
                     (append new-sccs `((SCC ,name ,(map eliminate-full-scan-gen-expr exprs))))]
                    [_ (displayln (format "expect a scc but got ~a" scc))]))
               ,execs)]))

; pass ra.1: split relation into binary join


; pass ra.2: mark relation used in SCC:
; TODO: move this eariler
; this pass will add 2 AST node GC and CHECK-FIXPOINT
; GC: all relations need be purged after SCC finished
; CHECK-FIXPOINT: relation used and modified in SCC
; this pass will also eliminate all GC/MODIFIED expr in AST
; and further flatten expression list, separate Fixopiont loop out

(define (mark-scc-rel scc)
  (match scc
    [`(SCC ,(? symbol? name) ,rules)
     `(SCC ,name
           ,`(RUN-ONCE ,@(all-once-rule-exprs rules))
           ,`(FIXPOINT-LOOP ,@(all-fixpoint-loop rules))
           ,`(FIXPOINT-CHECK ,@(all-modified-relation-in-exprs rules))
           ,`(GC ,@(all-gc-relation-in-exprs rules))
           ,`(IO ,@(all-io-in-exprs rules)))]))

(define (all-gc-relation-in-exprs exprs)
  (for/fold ([rels '()])
            ([expr (in-list exprs)])
    (match expr
      [`(GC ,(? q-mir-relation-ref? rel))
       (append rels (list (second rel)))]
      [_ rels])))
(define (all-modified-relation-in-exprs exprs)
  (for/fold ([rels '()])
            ([expr (in-list exprs)])
    (match expr
      [`(MODIFIED ,(? symbol? rel))
       (append rels (list rel))]
      [_ rels])))
(define (all-io-in-exprs exprs)
  (for/fold ([rels '()])
            ([expr (in-list exprs)])
    (match expr
      [`(IO ,(? symbol? rel) ,args)
       (append rels (list (cdr expr)))]
      [_ rels])))
(define (all-once-rule-exprs exprs)
  (for/fold ([rels '()])
            ([expr (in-list exprs)])
    (match expr
      [`(ONCE-RULE ,_ (,ops ...) ,_)
       (append rels (list expr))]
      [_ rels])))
(define (all-fixpoint-loop exprs)
  (define num-loop 0)
  (for/fold ([rels '()])
            ([expr (in-list exprs)])
    (match expr
      [`(FIXPOINT-LOOP (,rules ...))
       (set! num-loop (+ num-loop 1))
       (when (> num-loop 1)
         (displayln (format "expect only one fixpoint loop but got ~a !" expr)))
       (append rels (cdr expr))]
      [_ rels])))

(define (ra-mir-1->ra-mir-2 mir)
  (match mir
    [`(PROGRAM (,(? q-mir-relation? relations) ...)
               (, sccs ...)
               (,(? q-mir-exec? execs) ...))
     `(PROGRAM ,relations
               ,(for/fold ([new-sccs '()])
                          ([scc (in-list sccs)])
                  (match scc
                    [`(SCC ,(? symbol? name) (,exprs ...))
                     (append new-sccs (list (mark-scc-rel scc)))]
                    [_ (displayln (format "expect a scc but got ~a" scc))]))
               ,execs)]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; IR predicates

; q-mir
; a simplified version of souffle RAM IR, mostly flatten the nested structure
; (except for value expressions)

(define (q-mir-program? mir)
  (match mir
    [`(PROGRAM (,(? q-mir-relation? relations) ...)
               (,(? q-mir-scc? sccs) ...)
               (,(? q-mir-exec? execs) ...))
     #t]
    [_ #f]))

(define (q-mir-scc? mir)
  (define mir-scc-tag '(SCC))
  (match mir
    [`(SCC ,(? symbol? name) (,(? q-mir-expr? rules) ...)) #t]
    [_
     (when (member (car mir) mir-scc-tag)
       (displayln (format "expect a q-mir scc but got ~a" mir))) #f]))

(define (q-mir-expr? rule)
  (define mir-expr-tag '(ONCE-RULE RECURSIVE FIXPOINT-LOOP GC MODIFIED IO))
  (match rule
    [`(ONCE-RULE ,(? string? src-code) (,(? q-mir-operation? ops) ...) ,(? hash? var-bindings)) #t]
    [`(RECURSIVE ,(? string? src-code) (,(? q-mir-operation? ops) ...) ,(? hash? var-bindings)) #t]
    [`(FIXPOINT-LOOP (,(? q-mir-expr? exprs) ...)) #t]
    [`(GC ,(? q-mir-relation-ref?)) #t]
    [`(MODIFIED ,(? symbol? name)) #t]
    [`(IO ,rel-name ,args) #t]
    [_
     (when (member (car rule ) mir-expr-tag)
       (displayln (format "expect a q-mir expr but got ~a" rule)))
     #f]))

(define (q-mir-negate? mir)
  (match mir
    [`(NEGATE ,(? q-mir-relation-ref? rel)
              (,(? number? ungrounded-ids) ...)
              (,(? ram-expression? vars) ...)) #t]
    [_ #f]))

(define (q-mir-gen? mir)
  (match mir
    [`(GENERATE ,(? q-mir-relation-ref? rel) (,(? ram-expression? exprs) ...)) #t]
    [_ #f]))

(define (q-mir-operation? mir)
  (define mir-op-tag '(SCAN AGGREGATION NEGATE GENERATE FULL-SCAN))
  (match mir
    ; from cond
    [`(CONSTRAINT ,(? symbol? bin-op)
                  ,(? ram-expression? lhs) ,(? ram-expression? rhs)) #t]
    [(? q-mir-negate?) #t]
    [(? q-mir-gen?) #t]
    [`(AGGREGATION ,tuple-id ,agg ,condition ,index-op) #t]
    [`(SCAN ,(? q-mir-relation-ref? rel) ,tp (,(? ram-indices-comp? comps)  ...)) #t]
    [`(FULL-SCAN ,(? q-mir-relation-ref? rel) ,(? symbol? tuple-id)) #t]
    [_
     (when (member (car mir) mir-op-tag)
       (displayln (format "expect a q-mir operation but got ~a" mir)))
     #f]))

(define (q-mir-type? type)
  (match type
    [`INT #t]
    [`UNSIGNED #t]
    [`STRING #t]
    [(? extra-type?) #t]
    [_ #f]))

(define (q-mir-attr? attr)
  (match attr
    [`(ATTRIBUTE ,name ,type) #t]
    [_ (displayln (format "expect a q-mir relation attribute but got ~a" attr)) #f]))

(define (q-mir-relation? node)
  (match node
    [`(RELATION ,name (,(? q-mir-attr? attrs) ...)) #t]
    [_ (displayln (format "expect a q-mir relation but got ~a" node)) #f]))

(define (q-mir-relation-ref? ref)
  (match ref
    [`(REL ,(? symbol? name) ,(? symbol? type)) #t]
    [_ #f]))

(define (q-mir-exec? exec)
  (match exec
    [`(EXECUTE ,(? symbol? name)) #t]
    [_ #f]))


; ra-mir
; a middle level IR contains RA operators rather than query plans

(define (ra-mir-operation? mir)
  (match mir
    [`(COPY ,(? q-mir-relation-ref? src-rel) ,(? q-mir-relation-ref? dest-rel)
            ,(? symbol? tuple-id) (,(? ram-expression? exprs) ...)) #t]
    [_ #f]))

(define (ra-mir-scc? mir)
  (match mir
    [`(SCC ,(? symbol? name)
           (RUN-ONCE ,once-ops)
           (FIXPOINT-LOOP ,recursive-exprs)
           (FIXPOINT-CHECK ,need-check-rel-names)
           (GC ,gc-rel-names)
           (IO ,io-ops)) #t]
    [_ #f]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; main function, use IO port takes in a souffle RAM IR and output a q-mir IR
(define-values (in-ram-file out-q-mir-file)
  (command-line
   #:program "ram->q-mir"
   #:args (ram q-ir)
   (values ram q-ir)))

(define ram-f (open-input-file in-ram-file))
(define q-ir-f (open-output-file out-q-mir-file #:exists 'replace))

(define pass-map
  `(("flatten-nested-query" ,ram-program->q-mir)
    ("remove-newt-exist-check" ,q-mir->q-mir-2)
    ; ("remove-self-join-exist-check" ,q-mir-2->q-mir-3)
    ; ("full-scan-gen-to-copy" ,q-mir-3->ra-mir-0)
    ; ("further-flatten-scc" ,ra-mir-1->ra-mir-2)
    ))

; convert and write to output file
(call-with-values
 (lambda () (read ram-f))
 (lambda (ram)
   (define compiled-ir
     (for/fold ([res-ir ram])
               ([pass (in-list pass-map)])
       (match pass
         [`(,pass-name ,pass-f)
          (displayln (format "running pass: ~a" pass-name))
          (pass-f res-ir)])))
   (pretty-write
    compiled-ir
    q-ir-f)))
