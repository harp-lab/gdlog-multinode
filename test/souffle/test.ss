"+disconnected1() :-
    symbolic_operand(_,_,Dest,\"data\"),\n   
    Dest > 0x7FFE0000,\n   
    Dest < 0x7FFE1000.\n
    in file symbolization.dl [27:1-32:40]" 
(IF (AND (ISEMPTY +disconnected1) (NOT (ISEMPTY symbolic_operand)))
    (INDEXED_IF_EXISTS t0 symbolic_operand
        (INDEX  (= (TUPLE t0 3) (STRING "data")))
        (AND (AND (AND (CONSTRAINT <= (TUPLE t0 2)  (UNSIGNED 2147356672))
                       (CONSTRAINT != (TUPLE t0 2)  (UNSIGNED 2147356672)))
                  (CONSTRAINT != (TUPLE t0 2)  (UNSIGNED 2147352576)))
             (CONSTRAINT >= (TUPLE t0 2)  (UNSIGNED 2147352576)))
        (IF_BREAK (NOT (ISEMPTY +disconnected1)) 
           (INSERT () +disconnected1))))
