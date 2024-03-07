#include "../include/rt_incl.h"

void run(int argc, char *argv[], int block_size, int grid_size) {
ENVIRONMENT_INIT
DELCLARE_SCC(analysis)


// >>>>>>>>>>>>>>>>>>>>>>>>> DECLARE RELATION
DECLARE_RELATION_INPUT(analysis, assign, 2, 1);
ALIAS_RELATION(assign, INDEXED_REL(assign, 0_1, 1));
DECLARE_RELATION_INPUT(analysis, dereference, 2, 1);
ALIAS_RELATION(dereference, INDEXED_REL(dereference, 0_1, 1));
DECLARE_RELATION_OUTPUT(analysis, ValueAlias, 2, 1);
ALIAS_RELATION(ValueAlias, INDEXED_REL(ValueAlias, 0_1, 1));
DECLARE_RELATION_OUTPUT(analysis, ValueFlow, 2, 1);
ALIAS_RELATION(ValueFlow, INDEXED_REL(ValueFlow, 0_1, 1));
DECLARE_RELATION_OUTPUT(analysis, MemoryAlias, 2, 1);
ALIAS_RELATION(MemoryAlias, INDEXED_REL(MemoryAlias, 0_1, 1));

// >>>>>>>>>>>>>>>>>>>>>>>>> CREATE INDEX
CREATE_STATIC_INDEXED_RELATION(analysis, assign, 2, 1_0, 1);
// >>>>>>>>>>>>>>>>>>>>>>>>> CREATE TMP INDEX
CREATE_TMP_INDEXED_RELATION(analysis, ValueFlow, 2, 1_0, 1);


SCC_INIT(analysis);
MEMORY_STAT();
SCC_COMPUTE(analysis);
}

MAIN_ENTRANCE(run)
