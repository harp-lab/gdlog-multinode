#lang racket

; predicates for gdlog ir

(provide (all-defined-out))

(define (gdlog-program? node)
  (match node
    [`(PROGRAM ,(? gdlog-gpu-args? gpu-args)
        (,(? gdlog-rel-decl? rel-decls) ...)
        (,(? gdlog-input? inputs) ...)
        (,(? gdlog-stratum-ra-op? ops) ...)
        (,(? gdlog-output? out) ...)) #t]
    [_ (displayln "gdlog-program: not a program") #f]))

(define (gdlog-gpu-args? node)
  (match node
    [`(GPU_ARGS ,(? number? grid_size)
                ,(? number? block-size)) #t]
    ['DEFAULT_GPU_ARGS #t]
    [_ (displayln "gdlog-gpu-args: not a gpu-args") #f]))

(define (gdlog-rel-decl? node)
  (match node
    [`(REL_DECL ,(? string? rel-name) ,(? number? arity)
                ,(? string? index-column-size) ,(? number? depend-column-size)
                ,(? boolean? index-flag) ,(? boolean? tmp-flag)) #t]
    [_ (displayln "gdlog-rel-decl: not a rel-decl") #f]))

(define (gdlog-input? node)
    (match node
        [`(INPUT ,(? string? rel-name) ,(? string? file-name)) #t]
        [_ (displayln "gdlog-input: not an input") #f]))

(define (gdlog-stratum-relation-type? node)
  (member node '(NORMAL STATIC TMP)))

(define (gdlog-relation-ver? node)
  (member node '(FULL DELTA NEW)))

(define (gdlog-stratum-relation? node)
  (match node
    [`(STRATUM_RELATION ,(? string? rel-name) ,(? gdlog-stratum-relation-type? type)) #t]
    [_ (displayln "gdlog-stratum-relation: not a stratum-relation") #f]))


; (define (gdlog-hook-var? node)
;   (match node
;     [`(TUPLE_COLUMN ,(? symbol? tuple-id)) #t]
;     ))

; (define (gdlog-hook-instruction? instr)
;     (match instr
;       [`(CPP_RAW ,(? string? code))  #t]))

(define (gdlog-cpp-raw? node)  
  (match node
    [`(CPP_RAW ,(? string? code)) #t]
    [_ (displayln "gdlog-tuple-gen-hook: not a tuple-gen-hook") #f]))

(define (gdlog-stratum-ra-op? node)
  (match node
    [`(RA_JOIN ,(? symbol? inner-rel) ,(? gdlog-relation-ver? inner-ver)
               ,(? symbol? outer-rel) ,(? gdlog-relation-ver? outer-ver)
               ,(? symbol? output-rel) ,(? gdlog-cpp-raw? tp-gen)
               ,(? gdlog-cpp-raw? tp-pred)) #t]
    [`(RA_COPY ,(? symbol? src-rel) ,(? gdlog-relation-ver? src-ver)
               ,(? symbol? dest-rel) ,(? gdlog-cpp-raw? tp-pred)) #t]
    [`(RA_ACOPY ,(? symbol? src-rel) ,(? symbol? dest-rel)
                ,(? gdlog-cpp-raw? tp-pred)) #t]
    [_ (displayln "gdlog-stratum-ra-op: not a stratum-ra-op") #f]))


(define (gdlog-output? node)
  (match node
    [`(OUT ,relation_name ,port) #t]
    [_ (displayln "gdlog-output: not an output") #f]))

(define (gdlog-stratum? node)
  (match node
    [`(STRATUM (,(? gdlog-stratum-relation? rules) ...)
               (,(? gdlog-stratum-ra-op? ops) ...)) #t]
    [_ (displayln "gdlog-stratum: not a stratum") #f]))
