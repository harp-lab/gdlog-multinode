; transformed Souffle IR predicates
; please don't include paralle primitives in souffle ram IR
; please don't use debug statements in souffle ram IR

#lang racket

(provide (all-defined-out))

(define (ram-variable? node)
  (match node
    [`(VARIABLE ,name) #t]
    [_ #f]))

(define (ram-bool? node)
  (match node
    ['TRUE #t]
    ['FALSE #t]
    [_ #f]))

(define (ram-relation-attr? attr)
  (match attr
    [`(ATTRIBUTE ,(? symbol? name) ,(? symbol? type)) #t]
    [`(AUXILIARY ,(? symbol? name) ,(? symbol? type)) #t]
    [_  #f]))

(define (ram-relation? node)
  (match node
    [`(RELATION ,name (,(? ram-relation-attr? attrs) ...)
                ,rel-type) #t]
    [`(RELATION ,name (,(? ram-relation-attr? attrs) ...)) #t]
    [_ (displayln (format "expect a relation declaration but got ~a" node)) #f]))

(define (ram-relation-representation? node)
  (member node '(BRIE DEFAULT BTREE_DELETE EQREL INFO PROVENANCE BTREE)))

(define (ram-subroutine? sub)
  (match sub
    [`(SUBROUTINE ,(? symbol? name) ,(? ram-statement?)) #t]
    [`(SUBROUTINE ,(? symbol? name)) #t]
    [_ (displayln (format "expect a subroutine but got ~a" sub)) #f]))

(define (ram-program? node)
  (match node
    [`(PROGRAM (DECLARATION ,(? ram-relation? relations) ...)
               ( ,(? ram-subroutine? subroutines) ...)
               (MAIN ,(? ram-statement? main-stmts))) #t]
    [_ (displayln (format "expect a program but got ~a" node)) #f]))

(define (ram-indices-comp? comp)
  (match comp
    [`(= (TUPLE ,(? symbol? tuple-id) ,(? number? pos)) ,(? ram-expression? expr)) #t]
    [`(<= ,(? ram-expression? expr) (TUPLE ,(? symbol? tuple-id) ,(? number? pos))) #t]
    [`(<= (TUPLE ,(? symbol? tuple-id) ,(? number? pos)) ,(? ram-expression? expr)) #t]
    [`(<= ,(? ram-expression? expr1) (TUPLE ,(? symbol? tuple-id) ,(? number? pos))
          ,(? ram-expression? expr2)) #t]
    [_ #f]))
(define (ram-index-operation? node)
  (match node
    [`(INDEX ,(? ram-indices-comp? comps) ...) #t]
    [(? ram-indexed-aggregate?) #t]
    [`(INDEXED_IF_EXISTS ,(? symbol? tuple-id) ,(? symbol? rel-name)
                         ,(? ram-index-operation? index) ,(? ram-condition? r-cond)
                         ,(? ram-nested-operation? op)) #t]
    [`(FOR ,(? symbol? tuple-id) ,(? symbol? rel-name) ,(? ram-index-operation?)
           ,(? ram-nested-operation?)) #t]
    [_ #f]))


(define (ram-indexed-aggregate? node)
  (match node
    [`(INDEX_AGGREGATE (= (TUPLE ,tuple-id 0) ,(? ram-abstract-aggregation? agg))
                       ,(? symbol? tuple-id) ,(? symbol? rel-name)
                       ,(? ram-index-operation? indices) ,(? ram-condition? condition)
                       ,(? ram-nested-operation? index-op)) #t]
    [_ #f]))

(define (ram-abstract-aggregation? node)
  (match node
    [`(AGGREGATE_FUNCTION ,(? ram-aggregator? agg) ,(? ram-expression? expr)) #t]
    [(? ram-indexed-aggregate?) #t]
    [(? ram-aggregate?) #t]
    [(? symbol?) #t]
    [_  #f]))


(define (ram-if-exists? node)
  (match node
    [`(IF_EXISTS ,(? symbol? tuple-id) ,(? symbol? rel-name)
                 ,(? ram-condition? r-cond)
                 ,(? ram-nested-operation? r-op)) #t]
    [_ #f]))

(define (ram-aggregate? node)
  (match node
    [`(AGGREGATE ,(? symbol? tuple-id) ,(? ram-abstract-aggregation? agg)
                 ,_ ,_ ,(? symbol? rel-name) ,(? ram-condition? where-cond)
                 ,(? ram-operation? r-op)) #t]
    [_ #f]))

(define (ram-io? node)
  (define (ram-directive? directive)
    (match directive
      [`(= ,(? symbol? arg-name) ,(? string? arg-value)) #t]
      [_ #f]))
  (match node
    [`(IO ,(? symbol? rel-name) (,(? ram-directive? directives) ...)) #t]
    [_ #f]))

(define (ram-intrinsic-aggregator node)
  (member node '(COUNT SUM MIN MAX count sum min max)))

; add support for ProvenanceExistenceCheck
; add support for log
; add support for guard insert
; add support for autoinc

(define (ram-abstract-conditional? node)
  (match node
    [`(IF ,(? ram-condition? r-cond) ,(? ram-nested-operation? op)) #t]
    [`(IF_BREAK ,(? ram-condition? r-cond) ,(? ram-nested-operation? op)) #t]
    [_ #f]))

; TODO: change UNPACK
(define (ram-tuple-operation? node)
  (match node
    [`(UNPACKED_RECORD ,(? symbol? tuple-id) ,(? number? arity)
                       ,(? ram-expression? expr))
     #t]
    [`(UNPACK ,(? symbol? tuple-id) ,(? number? arity) 
              (,(? ram-expression? exprs) ...)
              ,(? ram-nested-operation? op)) #t]
    [`(INTRINSIC (,op ,(? ram-expression? args) ...)
                      ,(? symbol? tuple-id)) #t]
    [(? ram-relation-operation?) #t]
    [_ #f]))

(define (ram-nested-operation? node)
  (match node
    [`(INSERT (,(? ram-expression? exprs) ...) ,(? symbol? rel-name)) #t]
    [(? ram-tuple-operation?) #t]
    [(? ram-abstract-conditional?) #t]
    [_ #f]))

(define (ram-aggregator? node)
  (match node
    [(? ram-intrinsic-aggregator) #t]
    [_ #f]))

(define (ram-abstract-if-exists? node)
  (match node
    [(? ram-if-exists?) #t]
    [_ #f]))

(define (ram-relation-operation? node)
  (match node
    [`(FOR_IN ,(? symbol? tuple-id) ,(? symbol? rel-name) ,(? ram-operation? op))  #t]
    [(? ram-index-operation?) #t]
    [(? ram-if-exists?) #t]
    [(? ram-aggregate?) #t]
    [_ #f]))

(define (ram-bin-relation-statement? node)
  (match node
    [`(SWAP ,(? symbol? rel1) ,(? symbol? rel2) ) #t]
    [`(MERGE_EXTEND ,(? symbol? src-rel-name) ,(? symbol? dest-rel-name)) #t]
    [_ #f]))

(define (ram-condition? node)
  (match node
    [`(EXISTS ,(? symbol?) (,(? ram-expression?) ...)) #t]
    [`(NOT ,(? ram-condition? cond)) #t]
    [`(ISEMPTY ,(? symbol? rel-name)) #t]
    [`(CONSTRAINT ,(? symbol? bin-op)
                  ,(? ram-expression? lhs) ,(? ram-expression? rhs)) #t]
    [`(AND ,(? ram-condition? lhs) ,(? ram-condition? rhs)) #t]
    [(? ram-bool?) #t]
    [_ #f]))

(define (ram-operation? node)
  (match node
    [`(RETURN ,(? ram-expression? exprs) ...) #t]
    [`(ERASE (,(? ram-expression? exprs) ...) ,(? symbol? rel-name)) #t]
    [(? ram-nested-operation?) #t]
    [(? ram-relation-operation?) #t]
    [_ #f]))

(define(ram-statement? node)
  (match node
    [`(STMTS ,(? ram-statement?) ...) #t]
    [`(QUERY ,(? ram-operation?)) #t]
    [`(LOOP ,(? ram-statement? stmt)) #t]
    [`(CLEAR ,(? symbol? rel-name)) #t]
    [`(EXIT ,(? ram-condition? r-cond)) #t]
    [`(CALL ,(? symbol? subroutine-nam)) #t]
    [`(ASSIGN ,(? ram-variable? var) ,(? ram-expression? value)) #t]
    [`(DEBUG ,(? string? msg) ,(? ram-statement? stmt)) #t]
    [`(TIMER_ON ,(? symbol? rel-name) ,(? string? msg) ,(? ram-statement? stmt)) #t]
    [`(TIMER_ON ,(? symbol? rel-name) ,(? string? msg)) #t]
    [`(LOGSIZE ,rel-name ,debug-msg) #t]
    [(? ram-io?) #t]
    [(? ram-bin-relation-statement?) #t]
    [_ (displayln (format "expect a statement but got ~a" node)) #f]))

(define (ram-expression? node)
  (match node
    [`(USER_DEFINED_OPERATOR ,name ,(? ram-bool? stateful-flag)
                             ,(? (listof ram-expression?) args))
     #t]
    [`(INTRINSIC ,(? symbol? op) ,(? ram-expression? args) ...) #t]
    [`(INTRINSIC (,(? symbol? op) ,(? ram-expression? args) ...) ,tuple-id) #t]
    [`(USER_FUNCTOR ,(? string? func-name) ,stateful-flag (,(? ram-expression? exprs) ...)) #t]
    [`(UNSIGNED ,value) #t]
    [`(NUMBER ,(? number?)) #t]
    [`(FLOAT ,(? number?)) #t]
    ['UNDEF #t]
    [`(TUPLE ,(? symbol? identifier) ,(? number? element)) #t]
    [`(ARGUMENT ,(? number?)) #t]
    [`(STRING ,(? string?)) #t]
    [`(SIZE ,(? number?)) #t]
    [`(PACK ,(? ram-expression? exprs) ...) #t]
    [(? ram-variable?) #t]
    [_ (displayln (format "expect an expression but got ~a" node)) #f]))

(define (ram-node? node)
  (match node
    [(? ram-expression?) #t]
    [(? ram-operation?) #t]
    [(? ram-condition?) #t]
    [(? ram-statement?) #t]
    [(? ram-relation?) #t]
    [(? ram-program?) #t]
    [_ (displayln (format "expect a node but got ~a" node)) #f]))


;; for ddisasm
(define (extra-type? type)
  (member type '(STRING_NUM)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


; (define (gdlog-program? node)
;   (match node
;     [`(PROGRAM ,(? gdlog-gpu-args? gpu-args)
;                (,(? gdlog-rel-decl? rel-decls) ...)
;                (,(? gdlog-input? inputs) ...)
;                (,(? gdlog-stratum-ra-op? ops) ...)
;                (,(? gdlog-output? out) ...)) #t]
;     [_ (displayln "gdlog-program: not a program") #f]))

; (define (gdlog-gpu-args? node)
;   (match node
;     [`(GPU_ARGS ,(? number? grid_size)
;                 ,(? number? block-size)) #t]
;     [_ (displayln "gdlog-gpu-args: not a gpu-args") #f]))

; (define (gdlog-rel-decl? node)
;   (match node
;     [`(REL-DECL ,(? string? rel-name) ,(? number? arity)
;                 ,(? string? index-column-size) ,(? number? depend-column-size)
;                 ,(? boolean? index-flag) ,(? boolean? tmp-flag)) #t]
;     [_ (displayln "gdlog-rel-decl: not a rel-decl") #f]))

; (define (gdlog-input? node)
;   (match node
;     [`(INPUT ,(? string? rel-name) ,(? string? file-name)) #t]
;     [_ (displayln "gdlog-input: not an input") #f]))

; (define (gdlog-stratum-relation-type? node)
;   (member node '(NORMAL STATIC TMP)))

; (define (gdlog-relation-ver? node)
;   (member node '(FULL DELTA NEW)))

; (define (gdlog-stratum-relation? node)
;   (match node
;     [`(STRATUM_RELATION ,(? string? rel-name) ,(? gdlog-stratum-relation-type? type)) #t]
;     [_ (displayln "gdlog-stratum-relation: not a stratum-relation") #f]))


; ; (define (gdlog-hook-var? node)
; ;   (match node
; ;     [`(TUPLE_COLUMN ,(? symbol? tuple-id)) #t]
; ;     ))

; ; (define (gdlog-hook-instruction? instr)
; ;     (match instr
; ;       [`(CPP_RAW ,(? string? code))  #t]))

; (define (gdlog-cpp-raw? node)
;   (match node
;     [`(CPP_RAW ,(? string? code)) #t]
;     [_ (displayln "gdlog-tuple-gen-hook: not a tuple-gen-hook") #f]))

; (define (gdlog-stratum-ra-op? node)
;   (match node
;     [`(RA_JOIN ,(? symbol? inner-rel) ,(? gdlog-relation-ver? inner-ver)
;                ,(? symbol? outer-rel) ,(? gdlog-relation-ver? outer-ver)
;                ,(? symbol? output-rel) ,(? gdlog-cpp-raw? tp-gen)
;                ,(? gdlog-cpp-raw? tp-pred)) #t]
;     [`(RA_COPY ,(? symbol? src-rel) ,(? gdlog-relation-ver? src-ver)
;                ,(? symbol? dest-rel) ,(? gdlog-cpp-raw? tp-pred)) #t]
;     [`(RA_ACOPY ,(? symbol? src-rel) ,(? symbol? dest-rel)
;                 ,(? gdlog-cpp-raw? tp-pred)) #t]
;     [_ (displayln "gdlog-stratum-ra-op: not a stratum-ra-op") #f]))


; (define (gdlog-output? node)
;   (match node
;     [`(OUT ,relation_name ,port) #t]
;     [_ (displayln "gdlog-output: not an output") #f]))

; (define (gdlog-stratum? node)
;   (match node
;     [`(STRATUM (,(? gdlog-stratum-relation? rules) ...)
;                (,(? gdlog-stratum-ra-op? ops) ...)) #t]
;     [_ (displayln "gdlog-stratum: not a stratum") #f]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;





