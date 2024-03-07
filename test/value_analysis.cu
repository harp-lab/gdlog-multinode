
#include "../include/rt_incl.h"

void analysis_bench(int argc, char *argv[], int block_size, int grid_size) {
    ENVIRONMENT_INIT
    DELCLARE_SCC(analysis_scc)

    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, block_next, 3, 1);

    DECLARE_RELATION_INPUT(analysis_scc, block_last_instruction, 2, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, block_last_instruction, 2, 1_0,
                                   1)

    DECLARE_RELATION_INPUT(analysis_scc, direct_jump, 2, 1);

    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, jump_table_start, 5, 1)
    CREATE_FULL_INDEXED_RELATION(analysis_scc, jump_table_start, 3, 2_1_0, 2)

    // FIXME: remove test input for relative_jump_table_entry_candidate
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc,
                                  relative_jump_table_entry_candidate, 7, 1)
    CREATE_FULL_INDEXED_RELATION(
        analysis_scc, relative_jump_table_entry_candidate, 3, 1_2_4, 2)

    DECLARE_RELATION_OUTPUT(analysis_scc, jump_table_target, 2, 1);

    DECLARE_RELATION_OUTPUT(analysis_scc, cmp_defines, 3, 2);

    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, compare_and_jump_immediate, 5,
                                  1);

    DECLARE_RELATION_INPUT(analysis_scc, may_fallthrough, 2, 1);

    // FIXME: remove input later
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, flags_and_jump_pair, 3, 1);

    DECLARE_RELATION_INPUT(analysis_scc, instruction, 10, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, instruction, 2, 0_3, 1)

    DECLARE_RELATION_INPUT(analysis_scc, arch_cmp_operation, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, cmp_immediate_to_reg, 4, 1);
    DECLARE_RELATION_OUTPUT(analysis_scc_init, instruction_cmp_immediate_to_reg,
                            3, 1);

    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, compare_and_jump_register, 5,
                                  2);

    DECLARE_RELATION_INPUT(analysis_scc, cmp_reg_to_reg, 3, 1);

    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, value_reg, 7, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, reg_def_use_def_used, 4, 1);
    DECLARE_RELATION_OUTPUT(analysis_scc, const_value_reg_used, 5, 1);

    CREATE_FULL_INDEXED_RELATION(analysis_scc, value_reg, 6, 0_1_2_3_4_5, 2)
    CREATE_FULL_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 3, 0_1_2,
                                 2)

    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, def_used_for_address, 3, 1)
    DECLARE_RELATION_INPUT(analysis_scc, reg_used_for, 3, 2)

    // INITIALIZATION
    DECLARE_RELATION(analysis_scc, instruction_cmp_op, 1, 1);
    DECLARE_RELATION(analysis_scc, instruction_test_op, 1, 1);
    TMP_COPY(INIT_VER(analysis_scc), INDEXED_REL(instruction, 0_3, 1),
             instruction_cmp_op, COPY_END);
    TMP_CONST_FILTER(INIT_VER(analysis_scc), instruction_cmp_op,
                     {"ea", s2d("CMP")});
    TMP_COPY(INIT_VER(analysis_scc), INDEXED_REL(instruction, 0_3, 1),
             instruction_test_op, COPY_END);
    TMP_CONST_FILTER(INIT_VER(analysis_scc), instruction_test_op,
                     {"ea", s2d("TEST")});
    TMP_UNION(INIT_VER(analysis_scc), instruction_test_op, instruction_cmp_op);
    BINARY_JOIN(analysis_scc_init, cmp_immediate_to_reg, FULL,
                instruction_cmp_op, NEWT, instruction_cmp_immediate_to_reg,
                {"EA_cmp", "Reg", "_", "Immediate"}, {"EA_cmp", "_"},
                {"EA_cmp", "Reg", "Immediate"}, JOIN_END);

    // RECURSIVE SCC
    BINARY_JOIN(analysis_scc, INDEXED_REL(block_last_instruction, 1_0, 1), FULL,
                jump_table_target, DELTA, block_next, {"EA", "Block"},
                {"EA", "Dest"}, {"Block", "EA", "Dest"}, JOIN_END);

    SEMI_NAIVE_BINARY_JOIN(
        analysis_scc, INDEXED_REL(jump_table_start, 2_1_0, 2),
        INDEXED_REL(relative_jump_table_entry_candidate, 1_2_4, 2),
        jump_table_target, {"TableStart", "Size", "EA"},
        {"TableStart", "Size", "Dest"}, {"EA", "Dest"}, JOIN_END);

    DECLARE_TMP_COPY(analysis_scc, compare_and_jump_immediate, 3, 1, 1_2_3);
    DUPLICATE_TMP(analysis_scc,
                  INDEXED_REL(compare_and_jump_immediate, 1_2_3, 1), 3, 1,
                  compare_and_jump_immediate_1_NE_3)
    ALIAS_RELATION(INDEXED_REL(compare_and_jump_immediate, 1_2_3, 1),
                   compare_and_jump_immediate_1_E_3);
    TMP_CONST_FILTER(analysis_scc, compare_and_jump_immediate_1_E_3,
                     {"EA_jmp", "E", "Reg"});
    TMP_CONST_FILTER(analysis_scc, compare_and_jump_immediate_1_NE_3,
                     {"EA_jmp", "NE", "Reg"});
    BINARY_JOIN(analysis_scc, direct_jump, FULL,
                compare_and_jump_immediate_1_E_3, NEWT, cmp_defines,
                {"EA_jmp", "EA_dst"}, {"EA_jmp", "_", "Reg"},
                {"EA_jmp", "EA_dst", "Reg"}, JOIN_END);
    BINARY_JOIN(analysis_scc, may_fallthrough, FULL,
                compare_and_jump_immediate_1_NE_3, NEWT, cmp_defines,
                {"EA_jmp", "EA_dst"}, {"EA_jmp", "_", "Reg"},
                {"EA_jmp", "EA_dst", "Reg"}, JOIN_END);

    BINARY_JOIN(analysis_scc, instruction_cmp_immediate_to_reg, FULL,
                flags_and_jump_pair, DELTA, compare_and_jump_immediate,
                {"EA_cmp", "Reg", "Immediate"}, {"EA_cmp", "EA_jmp", "CC"},
                {"EA_cmp", "EA_jmp", "CC", "Reg", "Immediate"}, JOIN_END);

    BINARY_JOIN(analysis_scc, cmp_reg_to_reg, FULL, flags_and_jump_pair, DELTA,
                compare_and_jump_register, {"EA_cmp", "Reg1", "Reg2"},
                {"EA_cmp", "EA_jmp", "CC"},
                {"EA_cmp", "EA_jmp", "CC", "Reg1", "Reg2"}, JOIN_END);

    ALIAS_RELATION(INDEXED_REL(value_reg, 0_1_2_3_4_5, 2), value_reg_none_0);
    SEMI_NAIVE_BINARY_JOIN(
        analysis_scc, value_reg_none_0,
        INDEXED_REL(reg_def_use_def_used, 0_1_2, 2), const_value_reg_used,
        {"EARegDef", "Reg", "EADef", shashs("NONE"), snum(0), "Value"},
        {"EARegDef", "Reg", "UsedEA"},
        {"UsedEA", "EADef", "EARegDef", "Reg", "Value"}, JOIN_END);

    CREATE_TMP_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 3, 2_1_0,
                                2);
    BINARY_JOIN(analysis_scc, reg_used_for, FULL,
                INDEXED_REL(reg_def_use_def_used, 2_1_0, 2), NEWT,
                def_used_for_address, {"EA","Reg", "EA_def"},
                {"EA", "Reg", "Type"}, {"EA_def", "Reg", "Type"}, JOIN_END);

    BINARY_JOIN(analysis_scc, reg_used_for, FULL,
                INDEXED_REL(reg_def_use_def_used, 2_1_0, 2), NEWT,
                def_used_for_address, {"EA", "Reg", "Type"},
                {"EA", "Reg", "EA_def"}, {"EA_def", "Reg", "Type"}, JOIN_END);

    CREATE_FULL_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 3, 2_0_1,
                                 1)
    SEMI_NAIVE_BINARY_JOIN(
        analysis_scc, INDEXED_REL(reg_def_use_def_used, 2_0_1, 1),
        def_used_for_address, def_used_for_address,
        {"EA_used", "EA_def", "Reg"}, {"EA_used", "_", "Type"},
        {"EA_def", "Reg", "Type"}, JOIN_END);

    /////////////
    SYNC_INDEXED_RELATION(analysis_scc, jump_table_start, 2_1_0, 2);
    SYNC_INDEXED_RELATION(analysis_scc, relative_jump_table_entry_candidate,
                          1_2_4, 2);
    SYNC_INDEXED_RELATION(analysis_scc, value_reg, 0_1_2_3_4_5, 2);
    TMP_CONST_FILTER(
        analysis_scc, INDEXED_REL(value_reg, 0_1_2_3_4_5, 2),
        {"EARegDef", "Reg", "EADef", s2d("NONE"), n2d(0), "Value"});
    TMP_CONST_FILTER(
        analysis_scc_init, INDEXED_REL(value_reg, 0_1_2_3_4_5, 2),
        {"EARegDef", "Reg", "EADef", s2d("NONE"), n2d(0), "Value"});
    
    SYNC_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 0_1_2, 2);
    SYNC_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 2_0_1, 1);


    SCC_INIT(analysis_scc);
    MEMORY_STAT();
    SCC_COMPUTE(analysis_scc);
}

MAIN_ENTRANCE(analysis_bench)
