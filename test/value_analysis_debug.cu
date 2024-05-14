
#include "../include/rt_incl.h"

void run(int argc, char *argv[], int block_size, int grid_size) {
    ENVIRONMENT_INIT
    DELCLARE_SCC(analysis_scc)

    DECLARE_RELATION_INPUT(analysis_scc, reg_def_use_live_var_def, 4, 2);
    DECLARE_RELATION_INPUT(analysis_scc, stack_base_reg_move, 4, 2);
    DECLARE_RELATION_INPUT(analysis_scc, cmp_immediate_to_reg, 4, 1);
    DECLARE_RELATION_INPUT(analysis_scc, reg_def_use_block_last_def, 3, 2);
    DECLARE_RELATION_INPUT(analysis_scc, arch_register_size_bytes, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, code_in_block, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, is_xor_reset, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, binary_format, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, data_access, 8, 1);
    DECLARE_RELATION_INPUT(analysis_scc, defined_symbol, 9, 1);
    DECLARE_RELATION_INPUT(analysis_scc, direct_jump, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_reg_arithmetic_operation, 5, 2);
    DECLARE_RELATION_INPUT(analysis_scc, block_instruction_next, 3, 2);
    DECLARE_RELATION_INPUT(analysis_scc, reg_call, 2, 1);
    // DECLARE_RELATION_INPUT(analysis_scc, reg_def_use_used, 3, 2);
    DECLARE_RELATION_INPUT(analysis_scc, reg_map, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, stack_def_use_used_in_block, 5, 3);
    DECLARE_RELATION_INPUT(analysis_scc, stack_def_use_moves_limit, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_extend_load, 3, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_jump, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_memory_access, 9, 1);
    DECLARE_RELATION_INPUT(analysis_scc, instruction_has_relocation, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, cmp_reg_to_reg, 3, 1);
    DECLARE_RELATION_INPUT(analysis_scc, limit_reg_op, 4, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_extend_reg, 4, 2);
    DECLARE_RELATION_INPUT(analysis_scc, arch_conditional, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, op_immediate, 3, 1);
    DECLARE_RELATION_INPUT(analysis_scc, reg_jump, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_reg_reg_arithmetic_operation, 6,
                           2);
    DECLARE_RELATION_INPUT(analysis_scc, data_segment, 2, 0);
    DECLARE_RELATION_INPUT(analysis_scc, arch_condition_flags_reg, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_stack_pointer, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, reg_used_for, 3, 2);
    DECLARE_RELATION_INPUT(analysis_scc, limit_type_map, 5, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_cmp_operation, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, pc_relative_operand, 3, 1);
    DECLARE_RELATION_INPUT(analysis_scc, direct_call, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_move_reg_imm, 4, 2);
    DECLARE_RELATION_INPUT(analysis_scc, instruction_get_src_op, 3, 1);
    DECLARE_RELATION_INPUT(analysis_scc, adjusts_stack_in_block, 4, 2);
    DECLARE_RELATION_INPUT(analysis_scc, block_last_instruction, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, next, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, base_address, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, reg_def_use_flow_def, 4, 2);
    DECLARE_RELATION_INPUT(analysis_scc, op_regdirect, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, instruction_get_dest_op, 3, 1);
    DECLARE_RELATION_INPUT(analysis_scc, value_reg_functor, 7, 1);
    DECLARE_RELATION_INPUT(analysis_scc, stack_def_use_live_var_def, 6, 3);
    DECLARE_RELATION_INPUT(analysis_scc, reg_def_use_return_block_end, 4, 0);
    DECLARE_RELATION_INPUT(analysis_scc, track_register, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, instruction, 10, 1);
    DECLARE_RELATION_INPUT(analysis_scc, reg_def_use_used_in_block, 4, 2);
    DECLARE_RELATION_INPUT(analysis_scc, may_fallthrough, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, relative_jump_table_entry_candidate, 7,
                           2);
    DECLARE_RELATION_INPUT(analysis_scc, arch_move_reg_reg, 3, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_store_immediate, 8, 3);
    DECLARE_RELATION_INPUT(analysis_scc, arch_return_reg, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, symbolic_expr_from_relocation, 5, 1);
    DECLARE_RELATION_INPUT(analysis_scc, arch_call, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, simple_data_access_pattern, 4, 2);
    DECLARE_RELATION_INPUT(analysis_scc, stack_def_use_def, 3, 3);
    DECLARE_RELATION_INPUT(analysis_scc, no_value_reg_limit, 1, 1);
    DECLARE_RELATION_INPUT(analysis_scc, op_indirect, 7, 1);
    DECLARE_RELATION_INPUT(analysis_scc, instruction_get_op, 3, 1);
    DECLARE_RELATION_INPUT(analysis_scc, take_address, 2, 2);
    DECLARE_RELATION_INPUT(analysis_scc, op_regdirect_contains_reg, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, reg_def_use_defined_in_block, 2, 1);
    DECLARE_RELATION_INPUT(analysis_scc, conditional_jump, 1, 1);

    //////////////
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, cmp_defines, 3, 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init, "cmp_defines",
                   "cmp_defines__0_1_2__2");
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, compare_and_jump_indirect, 5,
                                  1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, jump_table_start, 5, 2);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc,
                                  stack_def_use_live_var_at_block_end, 4, 0);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, stack_def_use_live_var_used, 8,
                                  1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, value_reg_edge, 6, 2);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc,
                                  stack_def_use_live_var_at_prior_used, 4, 3);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, value_reg_unsupported, 2, 2);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc,
                                  reg_def_use_live_var_at_block_end, 3, 0);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc,
                                  reg_reg_arithmetic_operation_defs, 8, 2);
    // TODO: change to output only
    // DECLARE_RELATION_OUTPUT(analysis_scc, flags_and_jump_pair, 3, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, flags_and_jump_pair, 3, 1);
    // TODO: change to output only
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, jump_table_target, 2, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, last_value_reg_limit, 6, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, reg_def_use_return_val_used, 5,
                                  1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc,
                                  reg_def_use_live_var_at_prior_used, 3, 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init, "reg_def_use_live_var_def",
                   "reg_def_use_live_var_def__0_1_2_3__2");
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, const_value_reg_used, 5, 2);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, value_reg_limit, 5, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, reg_def_use_used, 3, 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init, "reg_def_use_used",
                   "reg_def_use_used__0_1_2__2");

    // ALIAS_RELATION(analysis_scc, analysis_scc_init, "value_reg_limit",
    //                "value_reg_limit__0_1_2_3_4__2");
    DECLARE_RELATION_OUTPUT(analysis_scc, jump_table_signed, 2, 1);

    //////////////

    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, compare_and_jump_register, 5,
                                  0);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, base_relative_jump, 2, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, reg_has_base_image, 2, 2);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, value_reg, 7, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, reg_def_use_live_var_used, 6,
                                  1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc,
                                  stack_def_use_live_var_used_in_block, 9, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, jump_table_element_access, 4,
                                  1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, reg_def_use_def_used, 4, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, stack_def_use_def_used, 7, 3);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, base_relative_operand, 3, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, block_next, 3, 1);
    // DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, compare_and_jump_immediate,
    // 5,
    //                               0);
    DECLARE_RELATION_OUTPUT(analysis_scc, compare_and_jump_immediate, 5, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, def_used_for_address, 3, 1);
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, got_relative_operand, 3, 1);

    CREATE_STATIC_INDEXED_RELATION(analysis_scc, take_address, 2, 0_1, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_def_use_live_var_def, 4,
                                   0_2_1_3, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, stack_base_reg_move, 3, 1_3_2,
                                   2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, cmp_immediate_to_reg, 3, 0_1_3,
                                   1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_stack_pointer, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_def_use_block_last_def, 3,
                                   0_2_1, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_def_use_block_last_def, 2,
                                   0_2, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_used_for, 3, 0_1_2, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_register_size_bytes, 2,
                                   0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, limit_type_map, 3, 0_1_3, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, limit_type_map, 3, 0_2_4, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, code_in_block, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, code_in_block, 2, 0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, code_in_block, 2, 0_1, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, is_xor_reset, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_cmp_operation, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_move_reg_imm, 3, 0_1_2,
                                   2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, direct_call, 2, 0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, data_access, 2, 0_7, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, instruction_get_src_op, 2, 0_2,
                                   1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, adjusts_stack_in_block, 3,
                                   1_2_3, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, block_last_instruction, 2, 1_0,
                                   1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, defined_symbol, 1, 8, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, direct_jump, 2, 0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, direct_jump, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, next, 2, 0_1, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_reg_arithmetic_operation,
                                   5, 0_2_1_3_4, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, block_instruction_next, 3,
                                   0_2_1, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_call, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, base_address, 1, 0, 0);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_def_use_flow_def, 3, 0_1_3,
                                   2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, op_regdirect, 2, 0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_map, 2, 0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, instruction_get_dest_op, 2,
                                   0_2, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, stack_def_use_used_in_block, 4,
                                   0_2_3_1, 3);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_jump, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_memory_access, 3, 1_0_4,
                                   1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, stack_def_use_live_var_def, 6,
                                   0_3_4_1_2_5, 3);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_def_use_return_block_end,
                                   2, 0_3, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, track_register, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, instruction, 2, 0_3, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, instruction, 5, 0_3_5_6_7, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_def_use_used_in_block, 3,
                                   0_2_1, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, instruction_has_relocation, 2,
                                   0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, may_fallthrough, 2, 0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, may_fallthrough, 2, 1_0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, may_fallthrough, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(
        analysis_scc, relative_jump_table_entry_candidate, 3, 1_2_4, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_move_reg_reg, 3, 0_1_2,
                                   1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, symbolic_expr_from_relocation,
                                   2, 0_4, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_call, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, cmp_reg_to_reg, 3, 0_1_2, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, limit_reg_op, 4, 2_0_1_3, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_extend_reg, 4, 0_1_2_3,
                                   2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_conditional, 2, 0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, op_immediate, 2, 0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, simple_data_access_pattern, 4,
                                   1_3_0_2, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, stack_def_use_def, 3, 0_1_2,
                                   3);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, op_indirect, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_jump, 1, 0, 1);
    CREATE_STATIC_INDEXED_RELATION(
        analysis_scc, arch_reg_reg_arithmetic_operation, 5, 0_2_1_4_5, 2);
    CREATE_STATIC_INDEXED_RELATION(
        analysis_scc, arch_reg_reg_arithmetic_operation, 6, 0_1_2_3_4_5, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, instruction_get_op, 2, 0_2, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, data_segment, 2, 0_1, 0);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, op_regdirect_contains_reg, 2,
                                   0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_extend_load, 3, 0_2_1, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, arch_extend_reg, 4, 0_1_3_2, 3)
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_map, 2, 0_1, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, limit_reg_op, 2, 0_2, 1);
    // CREATE_STATIC_INDEXED_RELATION(analysis_scc, limit_reg_op, 3, 2_0_3, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, limit_reg_op, 4, 0_2_1_3, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, reg_def_use_block_last_def, 3,
                                   0_1_2, 3);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, may_fallthrough, 2, 0_1, 2);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, limit_reg_op, 3, 2_0_1, 1);
    CREATE_STATIC_INDEXED_RELATION(analysis_scc, limit_reg_op, 2, 2_0, 1);

    CREATE_FULL_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 3, 0_1_2,
                                 2);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 3, 2_1_0,
                                 2);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 3, 2_0_1,
                                 1);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, def_used_for_address, 2, 0_2, 1);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, def_used_for_address, 2, 2_0, 1);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, stack_def_use_def_used, 5,
                                 3_1_2_4_5, 1);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 3, 1_0_2,
                                 1);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, reg_def_use_defined_in_block, 2,
                                 0_1, 2);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, value_reg_limit, 3, 0_1_2, 2);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, value_reg_limit, 5, 0_1_2_3_4,
                                 3);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, block_next, 3, 1_0_2, 2);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, block_next, 3, 0_1_2, 2);
    CREATE_FULL_INDEXED_RELATION(analysis_scc,
                                 reg_def_use_live_var_at_block_end, 3, 1_2_0, 2)

    analysis_scc.verbose_log = true;

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 1,
        {
            clause_meta("block_last_instruction", {"Block", "EA"}),
            clause_meta("jump_table_target", {"EA", "Dest"}),
        }, // -->
        clause_meta("block_next", {"Block", "EA", "Dest"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 2,
        {clause_meta("compare_and_jump_immediate",
                     {"_", "EA_jump", s2d("E"), "Reg", "_"}),
         clause_meta("direct_jump", {"EA_jump", "EA_dst"})}, // -->
        clause_meta("cmp_defines", {"EA_jump", "EA_dst", "Reg"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 3,
        {clause_meta("compare_and_jump_immediate",
                     {"_", "EA_jump", s2d("NE"), "Reg", "_"}),
         clause_meta("may_fallthrough", {"EA_jump", "EA_dst"})}, // -->
        clause_meta("cmp_defines", {"EA_jump", "EA_dst", "Reg"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 4,
        {clause_meta("flags_and_jump_pair", {"EA_cmp", "EA_jmp", "CC"}),
         clause_meta("instruction", {"EA_cmp", "_", "_", "Operation", "_", "_",
                                     "_", "_", "_", "_"}),
         clause_meta("arch_cmp_operation", {"Operation"}),
         clause_meta("cmp_immediate_to_reg",
                     {"EA_cmp", "Reg", "_", "Immediate"})},
        // -->
        clause_meta("compare_and_jump_immediate",
                    {"EA_cmp", "EA_jmp", "CC", "Reg", "Immediate"}),
        false);

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 5,
        {clause_meta("flags_and_jump_pair", {"EA_cmp", "EA_jmp", "CC"}),
         clause_meta("instruction", {"EA_cmp", "_", "_", "Operation", "_", "_",
                                     "_", "_", "_", "_"}),
         clause_meta("arch_cmp_operation", {"Operation"}),
         clause_meta("instruction_get_op", {"EA_cmp", "_", "IndirectOp"}),
         clause_meta("op_indirect",
                     {"IndirectOp", "_", "_", "_", "_", "_", "_"}),
         clause_meta("instruction_get_op", {"EA_cmp", "_", "ImmOp"}),
         clause_meta("op_immediate", {"ImmOp", "Immediate", "_", "_"})},
        // -->
        clause_meta("compare_and_jump_indirect",
                    {"EA_cmp", "EA_jmp", "CC", "IndirectOp", "Immediate"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 6,
        {clause_meta("flags_and_jump_pair", {"EA_cmp", "EA_jmp", "CC"}),
         clause_meta("cmp_reg_to_reg", {"EA_cmp", "Reg1", "Reg2"})},
        // -->
        clause_meta("compare_and_jump_register",
                    {"EA_cmp", "EA_jmp", "CC", "Reg1", "Reg2"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 7,
        {clause_meta("value_reg", {"EARegDef", "Reg", "EADef", s2d("NONE"),
                                   n2d(0), "Value", "_"}),
         clause_meta("reg_def_use_def_used",
                     {"EARegDef", "Reg", "UsedEA", "_"})},
        // -->
        clause_meta("const_value_reg_used",
                    {"UsedEA", "EADef", "EARegDef", "Reg", "Value"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 8,
        {clause_meta("reg_def_use_def_used", {"EA_def", "Reg", "EA", "_"}),
         clause_meta("reg_used_for", {"EA", "Reg", "Type"})},
        // -->
        clause_meta("def_used_for_address", {"EA_def", "Reg", "Type"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 9,
        {clause_meta("def_used_for_address", {"EA_used", "_", "Type"}),
         clause_meta("reg_def_use_def_used",
                     {"EA_def", "Reg", "EA_used", "_"})},
        // -->
        clause_meta("def_used_for_address", {"EA_def", "Reg", "Type"}));

    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 10,
    //     {clause_meta("def_used_for_address", {"EALoad", "Reg2", "Type"}),
    //      clause_meta("arch_memory_access",
    //                  {s2d("LOAD"), "EALoad", "_", "_", "Reg2", "RegBaseLoad",
    //                   s2d("NONE"), "_", "StackPosLoad"}),
    //      clause_meta("stack_def_use_def_used",
    //                  {"EAStore", "RegBaseStore", "StackPosStore", "EALoad",
    //                   "RegBaseLoad", "StackPosLoad", "_"}),
    //      clause_meta("arch_memory_access",
    //                  {s2d("STORE"), "EAStore", "_", "_", "Reg1",
    //                  "RegBaseStore",
    //                   s2d("NONE"), "_", "StackPosStore"}),
    //      clause_meta("reg_def_use_def_used",
    //                  {"EA_def", "Reg1", "EAStore", "_"})},
    //     // -->
    //     clause_meta("def_used_for_address", {"EA_def", "Reg1", "Type"}));
    // split this rule
    DECLARE_RELATION_OUTPUT(analysis_scc, def_used_for_address_load, 4, 3);
    DECLARE_RELATION_OUTPUT(analysis_scc, def_used_for_address_store, 5, 3);
    DECLARE_RELATION_OUTPUT(analysis_scc, def_used_for_address_load_store, 3,
                            2);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, def_used_for_address, 3, 0_1_2,
                                 2);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, stack_def_use_def_used, 6,
                                 0_1_2_3_4_5, 3);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, reg_def_use_def_used, 3, 1_2_0,
                                 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init, "def_used_for_address_load",
                   "def_used_for_address_load__0_1_2_3__3");
    ALIAS_RELATION(analysis_scc, analysis_scc_init,
                   "def_used_for_address_store",
                   "def_used_for_address_store__0_1_2_3_4__3");
    ALIAS_RELATION(analysis_scc, analysis_scc_init,
                   "def_used_for_address_load_store",
                   "def_used_for_address_load_store__0_1_2__2");
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 1001,
        {clause_meta("def_used_for_address", {"EALoad", "Reg2", "Type"}),
         clause_meta("arch_memory_access",
                     {s2d("LOAD"), "EALoad", "_", "_", "Reg2", "RegBaseLoad",
                      s2d("NONE"), "_", "StackPosLoad"})},
        // -->
        clause_meta("def_used_for_address_load",
                    {"EALoad", "RegBaseLoad", "StackPosLoad", "Type"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 1002,
        {clause_meta("stack_def_use_def_used",
                     {"EAStore", "RegBaseStore", "StackPosStore", "EALoad",
                      "RegBaseLoad", "StackPosLoad", "_"}),
         clause_meta("arch_memory_access",
                     {s2d("STORE"), "EAStore", "_", "_", "Reg1", "RegBaseStore",
                      s2d("NONE"), "_", "StackPosStore"})},
        // -->
        clause_meta(
            "def_used_for_address_store",
            {"EALoad", "RegBaseLoad", "StackPosLoad", "Reg1", "EAStore"}));
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 1003,
        {clause_meta("def_used_for_address_load",
                     {"EALoad", "RegBaseLoad", "StackPosLoad", "Type"}),
         clause_meta(
             "def_used_for_address_store",
             {"EALoad", "RegBaseLoad", "StackPosLoad", "Reg1", "EAStore"})},
        // -->
        clause_meta("def_used_for_address_load_store",
                    {"Reg1", "EAStore", "Type"}));
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 1004,
        {clause_meta("def_used_for_address_load_store",
                     {"Reg1", "EAStore", "Type"}),
         clause_meta("reg_def_use_def_used",
                     {"EA_def", "Reg1", "EAStore", "_"})},
        // -->
        clause_meta("def_used_for_address", {"EA_def", "Reg1", "Type"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 11,
        {clause_meta("arch_condition_flags_reg", {"Reg"}),
         clause_meta("reg_def_use_def_used",
                     {"EA_flags", "Reg", "EA_jmp", "_"}),
         clause_meta("arch_jump", {"EA_jmp"}),
         clause_meta("arch_conditional", {"EA_jmp", "CC"})},
        // -->
        clause_meta("flags_and_jump_pair", {"EA_flags", "EA_jmp", "CC"}));

    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 12,
    //     {clause_meta("pc_relative_operand", {"EA", n2d(1),
    //     "TableStartAddr"}),
    //      clause_meta("data_access",
    //                  {"EA", "_", "_", "_", "_", "_", "_", "Size"}),
    //      clause_meta("def_used_for_address", {"EA", "_", s2d("Jump")}),
    //      clause_meta("reg_def_use_def_used", {"EA", "Reg1", "EA_add", "_"}),
    //      clause_meta("reg_def_use_def_used", {"EA2", "Reg2", "EA_add", "_"}),
    //      clause_meta("take_address", {"EA2", "TableStartAddr"}),
    //      clause_meta("arch_reg_reg_arithmetic_operation",
    //                  {"EA_add", "_", "Reg2", "Reg1", n2d(1), n2d(0)}),
    //      clause_meta("data_segment", {"Beg", "End"}),
    //      clause_meta(">=", {"TableStartAddr", "Beg"}),
    //      clause_meta("<=", {"TableStartAddr", "End"})},
    //     // -->
    //     clause_meta("jump_table_element_access",
    //                 {"EA", "Size", "TableStartAddr", s2d("NONE")}));

    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 13,
    //     {clause_meta("pc_relative_operand", {"EA", n2d(1),
    //     "TableStartAddr"}),
    //      clause_meta("data_access",
    //                  {"EA", "_", "_", "_", "_", "_", "_", "Size"}),
    //      clause_meta("def_used_for_address", {"EA", "_", s2d("Call")}),
    //      clause_meta("reg_def_use_def_used", {"EA", "Reg1", "EA_add", "_"}),
    //      clause_meta("reg_def_use_def_used", {"EA2", "Reg2", "EA_add", "_"}),
    //      clause_meta("take_address", {"EA2", "TableStartAddr"}),
    //      clause_meta("arch_reg_reg_arithmetic_operation",
    //                  {"EA_add", "_", "Reg2", "Reg1", n2d(1), n2d(0)}),
    //      clause_meta("data_segment", {"Beg", "End"}),
    //      clause_meta(">=", {"TableStartAddr", "Beg"}),
    //      clause_meta("<=", {"TableStartAddr", "End"})},
    //     // -->
    //     clause_meta("jump_table_element_access",
    //                 {"EA", "Size", "TableStartAddr", s2d("NONE")}));

    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 14,
    //     {clause_meta("pc_relative_operand", {"EA", n2d(1),
    //     "TableStartAddr"}),
    //      clause_meta("data_access",
    //                  {"EA", "_", "_", "_", "_", "_", "_", "Size"}),
    //      clause_meta("def_used_for_address", {"EA", "_", s2d("Jump")}),
    //      clause_meta("reg_def_use_def_used", {"EA", "Reg1", "EA_add", "_"}),
    //      clause_meta("reg_def_use_def_used", {"EA2", "Reg2", "EA_add", "_"}),
    //      clause_meta("take_address", {"EA2", "TableStartAddr"}),
    //      clause_meta("arch_reg_reg_arithmetic_operation",
    //                  {"EA_add", "_", "Reg2", "Reg1", n2d(1), n2d(0)}),
    //      clause_meta("data_segment", {"Beg", "End"}),
    //      clause_meta(">=", {"TableStartAddr", "Beg"}),
    //      clause_meta("<=", {"TableStartAddr", "End"})},
    //     // -->
    //     clause_meta("jump_table_element_access",
    //                 {"EA", "Size", "TableStartAddr", s2d("NONE")}));

    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 15,
    //     {clause_meta("data_access", {"EA", "_", s2d("NONE"), "RegBase",
    //                                  "RegIndex", n2d(1), n2d(0), "Size"}),
    //      clause_meta("=/=", {"RegBase", s2d("NONE")}),
    //      clause_meta("=/=", {"RegIndex", s2d("NONE")}),
    //      clause_meta("const_value_reg_used",
    //                  {"EA", "_", "_", "RegIndex", "TableStart"}),
    //      clause_meta("data_segment", {"Beg", "End"}),
    //      clause_meta(">=", {"TableStart", "Beg"}),
    //      clause_meta("<=", {"TableStart", "End"})},
    //     // -->
    //     clause_meta("jump_table_element_access",
    //                 {"EA", "Size", "TableStart", "address", "RegBase"}));

    DECLARE_TMP_RELATION(analysis_scc, jump_table_element_access_tmp1, 4);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 1600,
        {
            clause_meta("data_access", {"EA", "_", s2d("NONE"), "RegBase",
                                        "RegIndex", n2d(1), n2d(0), "Size"}),
            clause_meta("=/=", {"RegBase", s2d("NONE")}),
            clause_meta("=/=", {"RegIndex", s2d("NONE")}),
            clause_meta("const_value_reg_used",
                        {"EA", "_", "_", "RegIndex", "TableStart"}),
            //  clause_meta("data_segment", {"Beg", "End"}),
            //  clause_meta(">=", {"TableStart", "Beg"}),
            //  clause_meta("<=", {"TableStart", "End"})
        },
        // -->
        clause_meta("jump_table_element_access_tmp1",
                    {"EA", "Size", "TableStart", "RegBase"}));
    auto cart_prod1_ft = TupleJoinFilter(
        2, 2, {BinaryFilterComparison::GE, BinaryFilterComparison::LE}, {4, 4},
        {0, 1});
    auto cart_prod1 = RelationalCartesian(
        data_segment, FULL, jump_table_element_access_tmp1, NEWT,
        jump_table_element_access, TupleGenerator(4, 2, {2, 3, 4, 5}),
        cart_prod1_ft, grid_size, block_size);
    cart_prod1.debug_flag = 0;
    analysis_scc.add_ra(cart_prod1);

    // 22812	0	104999	263382140
    DECLARE_TMP_RELATION(analysis_scc, jump_table_element_access_tmp2, 4);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 17,
        {
            clause_meta("data_access", {"EA", "_", s2d("NONE"), "RegBase",
                                        "RegIndex", "Size", "Offset", "Size"}),
            clause_meta("=/=", {"RegBase", s2d("NONE")}),
            clause_meta("=/=", {"RegIndex", s2d("NONE")}),
            clause_meta("const_value_reg_used",
                        {"EA", "_", "_", "RegBase", "Base"}),
            clause_meta("+", {"Base", "Offset"})
            //  clause_meta("data_segment", {"Beg", "End"}),
            //  clause_meta(">=", "Base", "Beg"),
            //  clause_meta("<=", "Base", "End")
        },
        // -->
        clause_meta("jump_table_element_access_tmp2",
                    {"EA", "Size", "Base", "RegIndex"}));
    auto cart_prod2 = RelationalCartesian(
        data_segment, FULL, jump_table_element_access_tmp2, NEWT,
        jump_table_element_access, TupleGenerator(4, 2, {2, 3, 4, 5}),
        cart_prod1_ft, grid_size, block_size);
    cart_prod2.debug_flag = 0;
    analysis_scc.add_ra(cart_prod2);

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 18,
        {
            clause_meta("jump_table_element_access",
                        {"EA", "Size", "TableStart", "_"}),
            clause_meta("*", {"Size", n2d(8)}),
            //  clause_meta("arch_extend_load", {"EA", "Signed", "_tmp_71"})},
            clause_meta("arch_extend_load", {"EA", "Signed", "Size"}),
            //  clause_meta("*", {"Size", n2d(8)})
        },
        // -->
        clause_meta("jump_table_signed", {"TableStart", "Signed"}));

    // TODO: untested rule
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 19,
        {clause_meta("jump_table_element_access",
                     {"EA", "Size", "TableStart", "_"}),
         clause_meta("value_reg", {"EA_used", "_", "EA", "Reg", "_", "_", "_"}),
         clause_meta("*", {"Size", n2d(8)}),
         clause_meta("arch_extend_reg", {"EA_used", "Reg", "Signed", "Size"})},
        // -->
        clause_meta("jump_table_signed", {"TableStart", "Signed"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 20,
        {clause_meta("jump_table_element_access",
                     {"EA", "_", "TableStart", "_"}),
         clause_meta("instruction_get_dest_op", {"EA", "_", "DstOp"}),
         clause_meta("op_regdirect", {"DstOp", "DefReg"}),
         clause_meta("reg_map", {"DefReg", "DefRegMapped"}),
         clause_meta("value_reg",
                     {"EA_used", "_", "EA", "DefRegMapped", "_", "_", "_"}),
         clause_meta("instruction_get_src_op", {"EA_used", "_", "Op"}),
         clause_meta("op_regdirect", {"Op", "UsedReg"}),
         clause_meta("reg_map", {"UsedReg", "DefRegMapped"}),
         clause_meta("arch_register_size_bytes", {"DefReg", "DefSize"}),
         clause_meta("arch_register_size_bytes", {"UsedReg", "UsedSize"}),
         clause_meta(">", {"UsedSize", "DefSize"})},
        // -->
        clause_meta("jump_table_signed", {"TableStart", n2d(0)}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 21,
        {clause_meta("jump_table_element_access",
                     {"_", n2d(8), "TableStart", "_"})}, // -->
        clause_meta("jump_table_signed", {"TableStart", n2d(1)}), true);

    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 22,
    //     {clause_meta("base_relative_jump", {"EA_base", "EA_jump"}),
    //      clause_meta("base_relative_operand", {"EA_base", "_", "Value"}),
    //      clause_meta("base_address", {"ImageBase"}),
    //      clause_meta("+", "ImageBase", "Value", "_tmp_1")},
    //     // -->
    //     clause_meta("jump_table_start", {"EA_jump", n2d(4), "_tmp_1",
    //     "ImageBase", n2d(1)}));

    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 23,
    //     {clause_meta("jump_table_element_access",
    //                  {"EA", "Size", "TableStart", "_"}),
    //      clause_meta("value_reg",
    //                  {"EA_add", "RegJump", "EA", "Reg", "Scale", "Base",
    //                  "_"}),
    //      clause_meta("=/=", {"Reg", s2d("NONE")}),
    //      clause_meta("reg_def_use_def_used",
    //                  {"EA_add", "RegJump", "EA_jump", "_"}),
    //      clause_meta("reg_call", {"EA_jump", "_"}),
    //      clause_meta("code_in_block", {"EA_jump", "_"})},
    //     // -->
    //     clause_meta("jump_table_start",
    //                 {"EA_jump", "Size", "TableStart", "Base", "Scale"}));
    // break into 2 join
    DECLARE_RELATION_OUTPUT(analysis_scc, jump_table_start_tmp1, 6, 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init, "jump_table_start_tmp1",
                   "jump_table_start_tmp1__0_1_2_3_4_5__2");
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 2300,
        {clause_meta("jump_table_element_access",
                     {"EA", "Size", "TableStart", "_"}),
         clause_meta("value_reg",
                     {"EA_add", "RegJump", "EA", "Reg", "Scale", "Base", "_"}),
         clause_meta("=/=", {"Reg", s2d("NONE")})},
        // -->
        clause_meta("jump_table_start_tmp1", {"EA_add", "RegJump", "Scale",
                                              "Base", "Size", "TableStart"}));
    DECLARE_RELATION_OUTPUT(analysis_scc, jump_table_start_tmp2, 3, 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init, "jump_table_start_tmp2",
                   "jump_table_start_tmp2__0_1_2__2");
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 2301,
        {clause_meta("reg_def_use_def_used",
                     {"EA_add", "RegJump", "EA_jump", "_"}),
         clause_meta("reg_call", {"EA_jump", "_"}),
         clause_meta("code_in_block", {"EA_jump", "_"})},
        // -->
        clause_meta("jump_table_start_tmp2", {"EA_add", "RegJump", "EA_jump"}));
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 2302,
        {clause_meta("jump_table_start_tmp1", {"EA_add", "RegJump", "Scale",
                                               "Base", "Size", "TableStart"}),
         clause_meta("jump_table_start_tmp2",
                     {"EA_add", "RegJump", "EA_jump"})},
        // -->
        clause_meta("jump_table_start",
                    {"EA_jump", "Size", "TableStart", "Base", "Scale"}));

    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 24,
    //     {clause_meta("jump_table_element_access",
    //                  {"EA", "Size", "TableStart", "_"}),
    //      clause_meta("value_reg",
    //                  {"EA_add", "RegJump", "EA", "Reg", "Scale", "Base",
    //                  "_"}),
    //      clause_meta("=/=", {"Reg", s2d("NONE")}),
    //      clause_meta("reg_def_use_def_used",
    //                  {"EA_add", "RegJump", "EA_jump", "_"}),
    //      clause_meta("reg_jump", {"EA_jump", "_"}),
    //      clause_meta("code_in_block", {"EA_jump", "_"})},
    //     // -->
    //     clause_meta("jump_table_start",
    //                 {"EA_jump", "Size", "TableStart", "Base", "Scale"}));
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 2400,
        {clause_meta("reg_def_use_def_used",
                     {"EA_add", "RegJump", "EA_jump", "_"}),
         clause_meta("reg_jump", {"EA_jump", "_"}),
         clause_meta("code_in_block", {"EA_jump", "_"})},
        // -->
        clause_meta("jump_table_start_tmp2", {"EA_add", "RegJump", "EA_jump"}));

    // TODO: not confirmed
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 25,
        {clause_meta("reg_jump", {"EA_jump", "_"}),
         clause_meta("code_in_block", {"EA_jump", "_"}),
         clause_meta("reg_def_use_def_used",
                     {"EA_base", "Reg", "EA_jump", "_"}),
         clause_meta("instruction", {"EA_base", "_", "_", s2d("ADD"), "_", "_",
                                     "_", "_", "_", "_"}),
         clause_meta("jump_table_element_access",
                     {"EA_base", "Size", "TableReference", "_"}),
         clause_meta("const_value_reg_used",
                     {"EA_base", "_", "_", "Reg", "TableReference"})},
        // -->
        clause_meta("jump_table_start", {"EA_jump", "Size", "TableReference",
                                         "TableReference", n2d(1)}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 26,
        {clause_meta("reg_jump", {"EA_jump", "_"}),
         clause_meta("code_in_block", {"EA_jump", "_"}),
         clause_meta("reg_def_use_def_used",
                     {"EA_base", "Reg", "EA_jump", "_"}),
         clause_meta("instruction", {"EA_base", "_", "_", s2d("SUB"), "_", "_",
                                     "_", "_", "_", "_"}),
         clause_meta("jump_table_element_access",
                     {"EA_base", "Size", "TableReference", "_"}),
         clause_meta("const_value_reg_used",
                     {"EA_base", "_", "_", "Reg", "TableReference"})},
        // -->
        clause_meta("jump_table_start",
                    {"EA_jump", "Size", "TableReference",
                     // FIXME: this is actually -1, but this column is not used,
                     // so we are safe in result
                     "TableReference", n2d(999)}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 27,
        {clause_meta("reg_jump", {"EA_jump", "_"}),
         clause_meta("code_in_block", {"EA_jump", "_"}),
         clause_meta("reg_def_use_def_used",
                     {"EA_base", "Reg", "EA_jump", "_"}),
         clause_meta("instruction", {"EA_base", "_", "_", s2d("ADD"), "_", "_",
                                     "_", "_", "_", "_"}),
         clause_meta("jump_table_element_access",
                     {"EA_base", "Size", "TableStart", "_"}),
         clause_meta("const_value_reg_used",
                     {"EA_base", "_", "_", "Reg", "TableReference"}),
         clause_meta("code_in_block", {"TableReference", "_"})},
        // -->
        clause_meta("jump_table_start", {"EA_jump", "Size", "TableStart",
                                         "TableReference", n2d(1)}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 29,
        {clause_meta("jump_table_start",
                     {"EA", "Size", "TableStart", "_", "_"}),
         clause_meta("relative_jump_table_entry_candidate",
                     {"_", "TableStart", "Size", "_", "Dest", "_", "_"})},
        // -->
        clause_meta("jump_table_target", {"EA", "Dest"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 30,
        {clause_meta("value_reg_limit",
                     {"From", "To", "Reg", "Value", "LimitType"})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"From", "To", "Reg", "Value", "LimitType", n2d(0)}));

    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 31,
    //     {clause_meta("last_value_reg_limit",
    //                  {"_", "EA", "PropagatedReg", "PropagatedVal",
    //                   "PropagatedType", "Steps"}),
    //      clause_meta("<=", {"Steps", n2d(3)}),
    //      clause_meta("code_in_block", {"EA", "Block"}),
    //      clause_meta("block_next", {"Block", "BlockEnd", "BlockNext"}),
    //      clause_meta("reg_def_use_defined_in_block", {"Block",
    //      "PropagatedReg"},
    //                  FULL, true),
    //      clause_meta("conditional_jump", {"BlockEnd"}, FULL, true)},
    //     // -->
    //     clause_meta("last_value_reg_limit",
    //                 {"BlockEnd", "BlockNext", "PropagatedReg",
    //                 "PropagatedVal",
    //                  "PropagatedType", "_tmp_1"}));

    DECLARE_RELATION_OUTPUT(analysis_scc, last_value_reg_limit_tmp0, 5, 1);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 3100,
        {clause_meta("last_value_reg_limit",
                     {"_", "EA", "PropagatedReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("<=", {"Steps", n2d(3)}),
         clause_meta("code_in_block", {"EA", "Block"})},
        // -->
        clause_meta("last_value_reg_limit_tmp0",
                    {"Block", "PropagatedReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));
    DECLARE_RELATION_OUTPUT(analysis_scc, last_value_reg_limit_tmp1, 6, 2);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 3101,
        {clause_meta("last_value_reg_limit_tmp0",
                     {"Block", "PropagatedReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("block_next", {"Block", "BlockEnd", "BlockNext"}),
         clause_meta("reg_def_use_defined_in_block", {"Block", "PropagatedReg"},
                     FULL, true),
         clause_meta("+", {"Steps", n2d(1)})},
        // -->
        clause_meta("last_value_reg_limit_tmp1",
                    {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 3102,
        {clause_meta("last_value_reg_limit_tmp0",
                     {"Block", "PropagatedReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("block_next", {"Block", "BlockEnd", "BlockNext"}),
         clause_meta("reg_def_use_defined_in_block", {"Block", "PropagatedReg"},
                     FULL, true),
         clause_meta("conditional_jump", {"BlockEnd"}, FULL, true),
         clause_meta("+", {"Steps", n2d(1)})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    // TODO: consider reuse last_value_reg_limit_tmp1 relation
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 32,
        {clause_meta("last_value_reg_limit_tmp0",
                     {"Block", "PropagatedReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("block_next", {"Block", "BlockEnd", "BlockNext"}),
         clause_meta("reg_def_use_defined_in_block", {"Block", "PropagatedReg"},
                     FULL, true),
         clause_meta("no_value_reg_limit", {"BlockEnd"}),
         clause_meta("+", {"Steps", n2d(1)})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    ALIAS_RELATION(analysis_scc, analysis_scc_init, "last_value_reg_limit_tmp1",
                   "last_value_reg_limit_tmp1__0_1_2_3_4_5__2");
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 33,
        {clause_meta("last_value_reg_limit_tmp1",
                     {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("cmp_defines", {"BlockEnd", "BlockNext", "Reg"}),
         clause_meta("=/=", {"Reg", "PropagatedReg"})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    CREATE_FULL_INDEXED_RELATION(analysis_scc, last_value_reg_limit_tmp1, 6,
                                 0_2_1_3_4_5, 2);
    CREATE_FULL_INDEXED_RELATION(analysis_scc, cmp_defines, 3, 0_2_1, 2);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 34,
        {clause_meta("last_value_reg_limit_tmp1",
                     {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("cmp_defines",
                     {"BlockEnd", "DefinedNext", "PropagatedReg"}),
         clause_meta("=/=", {"BlockNext", "DefinedNext"})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 35,
        {clause_meta("last_value_reg_limit_tmp1",
                     {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("value_reg_limit",
                     {"BlockEnd", "BlockNext", "Reg", "w1", "w2"}),
         clause_meta("=/=", {"Reg", "PropagatedReg"})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    CREATE_FULL_INDEXED_RELATION(analysis_scc, last_value_reg_limit_tmp1, 6,
                                 0_1_2_3_4_5, 3);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 36,
        {clause_meta("last_value_reg_limit_tmp1",
                     {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("value_reg_limit",
                     {"BlockEnd", "BlockNext", "PropagatedReg", "Val", "Type"}),
         clause_meta("=/=", {"PropagatedType", s2d("MAX")}),
         clause_meta("=/=", {"Type", s2d("MAX")}),
         clause_meta("<", {"PropagatedVal", "Val"})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 37,
        {clause_meta("last_value_reg_limit_tmp1",
                     {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("value_reg_limit",
                     {"BlockEnd", "BlockNext", "PropagatedReg", "Val", "Type"}),
         clause_meta("=/=", {"PropagatedType", s2d("MIN")}),
         clause_meta("=/=", {"Type", s2d("MIN")}),
         clause_meta(">", {"PropagatedVal", "Val"})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"BlockEnd", "BlockNext", "PropagatedReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    DECLARE_RELATION_OUTPUT(analysis_scc, last_value_reg_limit_tmp3, 6, 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init, "last_value_reg_limit_tmp3",
                   "last_value_reg_limit_tmp3__0_1_2_3_4_5__2");
    CREATE_FULL_INDEXED_RELATION(analysis_scc, last_value_reg_limit_tmp3, 6,
                                 1_0_2_3_4_5, 2);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 3800,
        {clause_meta(
             "last_value_reg_limit",
             {"_", "EA", "SrcReg", "PropagatedVal", "PropagatedType", "Steps"}),
         clause_meta("<=", {"Steps", n2d(3)}),
         clause_meta("code_in_block", {"EA", "Block"}),
         clause_meta("block_last_instruction", {"Block", "BlockEnd"}),
         clause_meta("limit_reg_op",
                     {"BlockEnd", "DstReg", "SrcReg", "Offset"}),
         clause_meta("<=", {"EA", "BlockEnd"}),
         clause_meta("+", {"PropagatedVal", "Offset"})},
        // -->
        clause_meta("last_value_reg_limit_tmp3",
                    {"Block", "BlockEnd", "DstReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 3801,
        {clause_meta("last_value_reg_limit_tmp3",
                     {"Block", "BlockEnd", "DstReg", "PropagatedVal",
                      "PropagatedType", "Steps"}),
         clause_meta("block_next", {"Block", "BlockEnd", "EA_next"}),
         clause_meta("code_in_block", {"BlockEnd", "Block"}),
         clause_meta("reg_def_use_block_last_def", {"BlockEnd", "_", "SrcReg"},
                     FULL, true),
         clause_meta("+", {"Steps", n2d(1)})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"BlockEnd", "EA_next", "DstReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    DECLARE_RELATION_OUTPUT(analysis_scc, last_value_reg_limit_tmp4, 8, 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init, "last_value_reg_limit_tmp4",
                   "last_value_reg_limit_tmp4__0_1_2_3_4_5_6_7__2");
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 3900,
        {clause_meta("last_value_reg_limit",
                     {"From", "EA", "SrcReg", "PropagatedVal", "PropagatedType",
                      "Steps"}),
         clause_meta("<=", {"Steps", n2d(3)}),
         clause_meta("code_in_block", {"EA", "Block"}),
         clause_meta("block_last_instruction", {"Block", "BlockEnd"}),
         clause_meta("limit_reg_op",
                     {"BlockEnd", "DstReg", "SrcReg", "Offset"}),
         clause_meta("<=", {"EA", "BlockEnd"}),
         clause_meta("+", {"PropagatedVal", "Offset"})},
        // -->
        clause_meta("last_value_reg_limit_tmp4",
                    {"BlockEnd", "Block", "DstReg", "PropagatedVal", "From",
                     "SrcReg", "PropagatedType", "Steps"}));
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 3901,
        {clause_meta("last_value_reg_limit_tmp4",
                     {"Block", "BlockEnd", "DstReg", "PropagatedVal", "From",
                      "SrcReg", "PropagatedType", "Steps"}),
         clause_meta("block_next", {"Block", "BlockEnd", "EA_next"}),
         clause_meta("code_in_block", {"BlockEnd", "Block"}),
         clause_meta("reg_def_use_block_last_def",
                     {"BlockEnd", "From", "SrcReg"}),
         clause_meta("+", {"Steps", n2d(1)})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"BlockEnd", "EA_next", "DstReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 40,
        {clause_meta("last_value_reg_limit",
                     {"From", "EA", "SrcReg", "PropagatedVal", "PropagatedType",
                      "Steps"}),
         clause_meta("<=", {"Steps", n2d(3)}),
         clause_meta("code_in_block", {"EA", "Block"}),
         clause_meta("block_last_instruction", {"Block", "BlockEnd"}),
         clause_meta("limit_reg_op", {"EA_mov", "DstReg", "SrcReg", "Offset"}),
         clause_meta("<=", {"EA", "EA_mov"}),
         clause_meta("<", {"EA_mov", "BlockEnd"}),
         clause_meta("+", {"PropagatedVal", "Offset"}),
         clause_meta("code_in_block", {"EA_mov", "Block"}),
         clause_meta("code_in_block", {"EA_mov", "inlined_Block_193"}),
         clause_meta("may_fallthrough", {"EA_mov", "EA_next"}),
         clause_meta("code_in_block", {"EA_next", "inlined_Block_193"}),
         clause_meta("reg_def_use_block_last_def",
                     {"EA_mov", "From", "SrcReg"}),
         clause_meta("+", {"Steps", n2d(1)})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"EA_mov", "EA_next", "DstReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 41,
        {clause_meta(
             "last_value_reg_limit",
             {"_", "EA", "SrcReg", "PropagatedVal", "PropagatedType", "Steps"}),
         clause_meta("<=", {"Steps", n2d(3)}),
         clause_meta("code_in_block", {"EA", "Block"}),
         clause_meta("block_last_instruction", {"Block", "BlockEnd"}),
         clause_meta("limit_reg_op", {"EA_mov", "DstReg", "SrcReg", "Offset"}),
         clause_meta("<=", {"EA", "EA_mov"}),
         clause_meta("<", {"EA_mov", "BlockEnd"}),
         clause_meta("+", {"PropagatedVal", "Offset"}),
         clause_meta("code_in_block", {"EA_mov", "Block"}),
         clause_meta("code_in_block", {"EA_mov", "inlined_Block_194"}),
         clause_meta("may_fallthrough", {"EA_mov", "EA_next"}),
         clause_meta("code_in_block", {"EA_next", "inlined_Block_194"}),
         clause_meta("reg_def_use_block_last_def", {"EA_mov", "_", "SrcReg"},
                     FULL, true),
         clause_meta("+", {"Steps", n2d(1)})},
        // -->
        clause_meta("last_value_reg_limit",
                    {"EA_mov", "EA_next", "DstReg", "PropagatedVal",
                     "PropagatedType", "Steps"}));

    // DECLARE_RELATION_OUTPUT(analysis_scc, last_value_reg_limit_tmp6, 3, 2);
    // DATALOG_RECURISVE_RULE(
    //     analysis_scc, analysis_scc_init, 4100,
    //     {clause_meta(
    //          "last_value_reg_limit",
    //          {"_", "EA", "SrcReg", "PropagatedVal", "PropagatedType",
    //          "Steps"}),
    //      clause_meta("<=", {"Steps", n2d(3)}),
    //      clause_meta("code_in_block", {"EA", "Block"}),
    //      clause_meta("block_last_instruction", {"Block", "BlockEnd"}),
    //      clause_meta("limit_reg_op", {"EA_mov", "DstReg", "SrcReg",
    //      "Offset"}), clause_meta("<=", {"EA", "EA_mov"}), clause_meta("<",
    //      {"EA_mov", "BlockEnd"}), clause_meta("+", {"PropagatedVal",
    //      "Offset"}), clause_meta("code_in_block", {"EA_mov", "Block"}),
    //      clause_meta("code_in_block", {"EA_mov", "inlined_Block_194"}),
    //      clause_meta("may_fallthrough", {"EA_mov", "EA_next"}),
    //      clause_meta("code_in_block", {"EA_next", "inlined_Block_194"}),
    //      clause_meta("reg_def_use_block_last_def", {"EA_mov", "_", "SrcReg"},
    //                  FULL, true),
    //      clause_meta("+", {"Steps", n2d(1)})},
    //     // -->
    //     clause_meta("last_value_reg_limit_tmp6",
    //                 {"BlockEnd", "EA_next", "DstReg", "PropagatedVal",
    //                  "PropagatedType", "Steps"}));

    FILTER_COPY(analysis_scc, "compare_and_jump_immediate", DELTA,
                "no_value_reg_limit", {"_", "EA_jump", s2d("O"), "_", "_"},
                {"EA_jump"});
    FILTER_COPY(analysis_scc, "compare_and_jump_immediate", DELTA,
                "no_value_reg_limit", {"_", "EA_jump", s2d("NO"), "_", "_"},
                {"EA_jump"});
    FILTER_COPY(analysis_scc, "compare_and_jump_immediate", DELTA,
                "no_value_reg_limit", {"_", "EA_jump", s2d("P"), "_", "_"},
                {"EA_jump"});
    FILTER_COPY(analysis_scc, "compare_and_jump_immediate", DELTA,
                "no_value_reg_limit", {"_", "EA_jump", s2d("PE"), "_", "_"},
                {"EA_jump"});
    FILTER_COPY(analysis_scc, "compare_and_jump_immediate", DELTA,
                "no_value_reg_limit", {"_", "EA_jump", s2d("S"), "_", "_"},
                {"EA_jump"});
    FILTER_COPY(analysis_scc, "compare_and_jump_immediate", DELTA,
                "no_value_reg_limit", {"_", "EA_jump", s2d("NS"), "_", "_"},
                {"EA_jump"});

    DECLARE_RELATION_OUTPUT(analysis_scc, no_value_reg_limit_debug, 1, 1);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 42,
        {clause_meta("compare_and_jump_register",
                     {"EA_cmp", "EA_jmp", "_", "Reg1", "Reg2"}),
         clause_meta("reg_def_use_block_last_def", {"EA_cmp", "_", "Reg2"},
                     FULL, true),
         clause_meta("reg_def_use_block_last_def", {"EA_cmp", "EA", "Reg1"}),
         clause_meta("arch_move_reg_imm", {"EA", "Reg1", "_", "_"}, FULL,
                     true)},
        // -->
        clause_meta("no_value_reg_limit", {"EA_jmp"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 43,
        {clause_meta("compare_and_jump_register",
                     {"EA_cmp", "EA_jmp", "_", "Reg1", "Reg2"}),
         clause_meta("reg_def_use_block_last_def", {"EA_cmp", "_", "Reg1"},
                     FULL, true),
         clause_meta("reg_def_use_block_last_def", {"EA_cmp", "EA", "Reg2"}),
         clause_meta("arch_move_reg_imm", {"EA", "Reg2", "_", "_"}, FULL,
                     true)},
        // -->
        clause_meta("no_value_reg_limit", {"EA_jmp"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 44,
        {clause_meta("compare_and_jump_register",
                     {"EA_cmp", "EA_jmp", "_", "Reg1", "Reg2"}),
         clause_meta("reg_def_use_block_last_def", {"EA_cmp", "_", "Reg2"}),
         clause_meta("reg_def_use_block_last_def", {"EA_cmp", "EA", "Reg1"}),
         clause_meta("arch_move_reg_imm", {"EA", "Reg1", "_", "_"}, FULL, true),
         clause_meta("arch_move_reg_imm", {"EA", "Reg2", "_", "_"}, FULL,
                     true)},
        // -->
        clause_meta("no_value_reg_limit", {"EA_jmp"}));

    // no_value_reg_limit_debug(EA_jmp),
    // no_value_reg_limit(EA_jmp) :-
    //    flags_and_jump_pair(EA_cmp,EA_jmp,_),
    //    instruction(EA_cmp,_,_,Operation,_,_,_,_,_,_),
    //    !arch_cmp_operation(Operation).
    // DECLARE_RELATION_OUTPUT(analysis_scc, instruction_non_cmp, 1, 1);
    DECLARE_RELATION(analysis_scc, instruction_non_cmp, 1, 1);
    analysis_scc_init.add_relations(instruction_non_cmp, false);
    NEGATE(analysis_scc_init, analysis_scc_init, "arch_cmp_operation", FULL,
           "instruction", FULL, "instruction_non_cmp", {"Operation"},
           {"EA_cmp", "_", "_", "Operation", "_", "_", "_", "_", "_", "_"},
           {"EA_cmp"}, false, false);
    analysis_scc.add_relations(instruction_non_cmp, true);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 45,
        {clause_meta("flags_and_jump_pair", {"EA_cmp", "EA_jmp", "_"}),
         clause_meta("instruction_non_cmp", {"EA_cmp"})},
        // -->
        clause_meta("no_value_reg_limit", {"EA_jmp"}));

    DECLARE_RELATION_OUTPUT(analysis_scc, reg_def_use_def_used_debug, 4, 1);
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 46,
        {clause_meta("reg_def_use_used", {"EA_used", "Var", "Index"}),
         clause_meta("reg_def_use_block_last_def",
                     {"EA_used", "EA_def", "Var"})},
        // -->
        clause_meta("reg_def_use_def_used",
                    {"EA_def", "Var", "EA_used", "Index"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 47,
        {clause_meta("reg_def_use_live_var_at_block_end",
                     {"Block", "BlockUsed", "Var"}),
         clause_meta("reg_def_use_live_var_def",
                     {"Block", "VarIdentity", "Var", "EA_def"}),
         clause_meta("reg_def_use_live_var_used",
                     {"BlockUsed", "Var", "Var", "EA_used", "Index", "_"})},
        // -->
        clause_meta("reg_def_use_def_used",
                    {"EA_def", "VarIdentity", "EA_used", "Index"}));

    // reg_def_use_def_used(EA_def,Var,Next_EA_used,NextIndex) :-
    //    reg_def_use_live_var_at_prior_used(EA_used,NextUsedBlock,Var),
    //    reg_def_use_def_used(EA_def,Var,EA_used,_),
    //    reg_def_use_live_var_used(NextUsedBlock,Var,Var,Next_EA_used,NextIndex,_).
    DECLARE_RELATION_OUTPUT(analysis_scc, reg_def_use_live_var_used_tmp1, 4, 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init,
                   "reg_def_use_live_var_used_tmp1",
                   "reg_def_use_live_var_used_tmp1__0_1_2_3__2");
    analysis_scc_init.add_relations(reg_def_use_live_var_used_tmp1, false);
    FILTER_COPY(
        analysis_scc_init, "reg_def_use_live_var_used", FULL,
        "reg_def_use_live_var_used_tmp1",
        {"NextUsedBlock", "Var", "Var", "Next_EA_used", "NextIndex", "_"},
        {"NextUsedBlock", "Var", "Next_EA_used", "NextIndex"});
    FILTER_COPY(
        analysis_scc, "reg_def_use_live_var_used", DELTA,
        "reg_def_use_live_var_used_tmp1",
        {"NextUsedBlock", "Var", "Var", "Next_EA_used", "NextIndex", "_"},
        {"NextUsedBlock", "Var", "Next_EA_used", "NextIndex"});
    CREATE_FULL_INDEXED_RELATION(
        analysis_scc, reg_def_use_live_var_at_prior_used, 3, 1_2_0, 2);
    DECLARE_RELATION_OUTPUT(analysis_scc, reg_def_use_def_used_tmp2, 4, 2);
    ALIAS_RELATION(analysis_scc, analysis_scc_init, "reg_def_use_def_used_tmp2",
                   "reg_def_use_def_used_tmp2__0_1_2_3__2");
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 4800,
        {clause_meta("reg_def_use_live_var_used_tmp1",
                     {"NextUsedBlock", "Var", "Next_EA_used", "NextIndex"}),
         clause_meta("reg_def_use_live_var_at_prior_used",
                     {"EA_used", "NextUsedBlock", "Var"})},
        // -->
        clause_meta("reg_def_use_def_used_tmp2",
                    {"EA_used", "Var", "Next_EA_used", "NextIndex"}));
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 4801,
        {clause_meta("reg_def_use_def_used_tmp2",
                     {"EA_used", "Var", "Next_EA_used", "NextIndex"}),
         clause_meta("reg_def_use_def_used",
                     {"EA_def", "Var", "EA_used", "_"})},
        // -->
        clause_meta("reg_def_use_def_used",
                    {"EA_def", "Var", "Next_EA_used", "NextIndex"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 49,
        {clause_meta("reg_def_use_return_val_used",
                     {"_", "Callee", "Reg", "EA_used", "Index"}),
         clause_meta("reg_def_use_return_block_end",
                     {"Callee", "_", "_", "BlockEnd"}),
         clause_meta("reg_def_use_block_last_def",
                     {"BlockEnd", "EA_def", "Reg"})},
        // -->
        clause_meta("reg_def_use_def_used",
                    {"EA_def", "Reg", "EA_used", "Index"}));

    // reg_def_use_return_val_used(EA_call,Callee,Reg,EA_used,Index_used) :-
    //    arch_return_reg(Reg),
    //    reg_def_use_def_used(EA_call,Reg,EA_used,Index_used),
    //    direct_call(EA_call,Callee).
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 50,
        {clause_meta("reg_def_use_def_used",
                     {"EA_call", "Reg", "EA_used", "Index_used"}),
         clause_meta("arch_return_reg", {"Reg"}),
         clause_meta("direct_call", {"EA_call", "Callee"})},
        // -->
        clause_meta("reg_def_use_return_val_used",
                    {"EA_call", "Callee", "Reg", "EA_used", "Index_used"}));

    // DATALOG_RECURSIVE_RULE(
    //     analysis_scc, analysis_scc_init, 48,
    //     {clause_meta(
    //          "reg_def_use_live_var_used_tmp1",
    //          {"NextUsedBlock", "Var", "Next_EA_used", "NextIndex"}),
    //      clause_meta("reg_def_use_live_var_at_prior_used",
    //                  {"EA_used", "NextUsedBlock", "Var"}),
    //      clause_meta("reg_def_use_def_used",
    //                  {"EA_def", "Var", "EA_used", "_"})},
    //     // -->
    //     clause_meta("reg_def_use_def_used_debug",
    //                 {"EA_def", "Var", "Next_EA_used", "NextIndex"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 100,
        {clause_meta("compare_and_jump_immediate",
                     {"_", "EA_jmp", "CC", "Reg", "Immediate"}),
         clause_meta("track_register", {"Reg"}),
         clause_meta("may_fallthrough", {"EA_jmp", "_"}),
         clause_meta("limit_type_map",
                     {"CC", "BranchLT", "_", "BranchOffset", "_"}),
         clause_meta("direct_jump", {"EA_jmp", "EA_branch"}),
         clause_meta("+", {"Immediate", "BranchOffset"})},
        // -->
        clause_meta("value_reg_limit",
                    {"EA_jmp", "EA_branch", "Reg", "Immediate", "BranchLT"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 101,
        {clause_meta("compare_and_jump_immediate",
                     {"_", "EA_jmp", "CC", "Reg", "Immediate"}),
         clause_meta("track_register", {"Reg"}),
         clause_meta("direct_jump", {"EA_jmp", "_"}),
         clause_meta("limit_type_map",
                     {"CC", "_", "FallthroughLT", "_", "FallthroughOffset"}),
         clause_meta("may_fallthrough", {"EA_jmp", "EA_fallthrough"}),
         clause_meta("+", {"Immediate", "FallthroughOffset"})},
        // -->
        clause_meta("value_reg_limit", {"EA_jmp", "EA_fallthrough", "Reg",
                                        "Immediate", "FallthroughLT"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 102,
        {clause_meta("compare_and_jump_register",
                     {"EA_cmp", "EA_jmp", "CC", "Reg1", "Reg2"}),
         clause_meta("may_fallthrough", {"EA_jmp", "_"}),
         clause_meta("track_register", {"Reg2"}),
         clause_meta("limit_type_map", {"CC", "LT1", "_", "Offset1", "_"}),
         clause_meta("reg_def_use_block_last_def",
                     {"EA_cmp", "EA_regdef", "Reg1"}),
         clause_meta("arch_move_reg_imm", {"EA_regdef", "Reg1", "Immediate", "_"}),
         clause_meta("direct_jump", {"EA_jmp", "EA_branch"}),
         clause_meta("+", {"Immediate", "Offset1"})},
        // -->
        clause_meta("value_reg_limit",
                    {"EA_jmp", "EA_branch", "Reg2", "Immediate", "LT1"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 103,
        {clause_meta("compare_and_jump_register",
                     {"EA_cmp", "EA_jmp", "CC", "Reg1", "Reg2"}),
         clause_meta("may_fallthrough", {"EA_jmp", "_"}),
         clause_meta("track_register", {"Reg1"}),
         clause_meta("limit_type_map", {"CC", "_", "LT2", "Offset2", "_"}),
         clause_meta("reg_def_use_block_last_def",
                     {"EA_cmp", "EA_regdef", "Reg2"}),
         clause_meta("arch_move_reg_imm", {"EA_regdef", "Reg2", "Immediate", "_"}),
         clause_meta("direct_jump", {"EA_jmp", "EA_branch"}),
         clause_meta("+", {"Immediate", "Offset2"})},
        // -->
        clause_meta("value_reg_limit",
                    {"EA_jmp", "EA_branch", "Reg1", "Immediate", "LT2"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 104,
        {clause_meta("compare_and_jump_register",
                     {"EA_cmp", "EA_jmp", "CC", "Reg1", "Reg2"}),
         clause_meta("direct_jump", {"EA_jmp", "_"}),
         clause_meta("track_register", {"Reg2"}),
         clause_meta("limit_type_map", {"CC", "_", "LT2", "Offset2", "_"}),
         clause_meta("reg_def_use_block_last_def",
                     {"EA_cmp", "EA_regdef", "Reg1"}),
         clause_meta("arch_move_reg_imm", {"EA_regdef", "Reg1", "Immediate", "_"}),
         clause_meta("may_fallthrough", {"EA_jmp", "EA_fallthrough"}),
         clause_meta("+", {"Immediate", "Offset2"})},
        // -->
        clause_meta("value_reg_limit",
                    {"EA_jmp", "EA_fallthrough", "Reg2", "Immediate", "LT2"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 105,
        {clause_meta("compare_and_jump_register",
                     {"EA_cmp", "EA_jmp", "CC", "Reg1", "Reg2"}),
         clause_meta("direct_jump", {"EA_jmp", "_"}),
         clause_meta("track_register", {"Reg1"}),
         clause_meta("reg_def_use_block_last_def",
                     {"EA_cmp", "EA_regdef", "Reg2"}),
         clause_meta("arch_move_reg_imm", {"EA_regdef", "Reg2", "Immediate", "_"}),
         clause_meta("may_fallthrough", {"EA_jmp", "EA_fallthrough"}),
         clause_meta("limit_type_map", {"CC", "LT1", "_", "Offset1", "_"}),
         clause_meta("+", {"Immediate", "Offset1"})},
        // -->
        clause_meta("value_reg_limit",
                    {"EA_jmp", "EA_fallthrough", "Reg1", "Immediate", "LT1"}));


    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 106,
        {clause_meta("compare_and_jump_indirect",
                     {"EA_cmp", "EA_jmp", "CC", "IndirectOp", "Immediate"}),
         clause_meta("next", {"EA_cmp", "EA_jmp"}),
         clause_meta("limit_type_map", {"CC", "BranchLT", "_", "BranchOffset", "_"}),
         clause_meta("direct_jump", {"EA_jmp", "EA_target"}),
         clause_meta("arch_memory_access",
                     {s2d("LOAD"), "EA_target", "_", "_", "Reg", "_", "_", "_", "_"}),
         clause_meta("track_register", {"Reg"}),
         clause_meta("instruction_get_op", {"EA_target", "_", "IndirectOp"}),
         clause_meta("code_in_block", {"EA_target", "inlined_Block_887"}),
         clause_meta("may_fallthrough", {"EA_target", "EA_limited"}),
         clause_meta("code_in_block", {"EA_limited", "inlined_Block_887"}),
         clause_meta("+", {"Immediate", "BranchOffset"})},
        // -->
        clause_meta("value_reg_limit",
                    {"EA_target", "EA_limited", "Reg", "Immediate", "BranchLT"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 107,
        {clause_meta("compare_and_jump_indirect",
                     {"EA_cmp", "EA_jmp", "CC", "IndirectOp", "Immediate"}),
         clause_meta("next", {"EA_cmp", "EA_jmp"}),
         clause_meta("limit_type_map", {"CC", "_", "FallthroughLT", "_", "FallthroughOffset"}),
         clause_meta("may_fallthrough", {"EA_jmp", "EA_target"}),
         clause_meta("arch_memory_access",
                     {s2d("LOAD"), "EA_target", "_", "_", "Reg", "_", "_", "_", "_"}),
         clause_meta("track_register", {"Reg"}),
         clause_meta("instruction_get_op", {"EA_target", "_", "IndirectOp"}),
         clause_meta("code_in_block", {"EA_target", "inlined_Block_888"}),
         clause_meta("may_fallthrough", {"EA_target", "EA_limited"}),
         clause_meta("code_in_block", {"EA_limited", "inlined_Block_888"}),
         clause_meta("+", {"Immediate", "FallthroughOffset"})},
        // -->
        clause_meta("value_reg_limit",
                    {"EA_target", "EA_limited", "Reg", "Immediate", "FallthroughLT"}));
    
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 108,
        {clause_meta("def_used_for_address", {"EA", "Reg", "_"}),
         clause_meta("arch_move_reg_reg", {"EA", "Dst", "Src"}),
         clause_meta("track_register", {"Dst"}),
         clause_meta("track_register", {"Src"}, FULL, true)},
        // -->
        clause_meta("value_reg_unsupported", {"EA", "Reg"}));

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 109,
        {clause_meta("def_used_for_address", {"EA", "Reg", "_"}),
         clause_meta("arch_call", {"EA", "_"})},
        // -->
        clause_meta("value_reg_unsupported", {"EA", "Reg"}));

    // value_reg_unsupported(EA,Reg) :-
    //    def_used_for_address(EA,Reg,_),
    //    arch_memory_access("LOAD",EA,_,_,Reg,RegBase,_,_,_),
    //    RegBase != "NONE".

    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 110,
        {clause_meta("def_used_for_address", {"EA", "Reg", "_"}),
         clause_meta("arch_memory_access",
                     {"LOAD", "EA", "_", "_", "Reg", "RegBase", "_", "_", "_"}),
         clause_meta("=/=", {"RegBase", s2d("NONE")})},
        // -->
        clause_meta("value_reg_unsupported", {"EA", "Reg"}));

    // value_reg_unsupported(EA,Reg) :-
    //    def_used_for_address(EA,Reg,_),
    //    arch_memory_access("LOAD",EA,_,_,Reg,_,RegIndex,_,_),
    //    RegIndex != "NONE".
    DATALOG_RECURSIVE_RULE(
        analysis_scc, analysis_scc_init, 111,
        {clause_meta("def_used_for_address", {"EA", "Reg", "_"}),
         clause_meta("arch_memory_access", {"LOAD", "EA", "_", "_", "Reg", "_",
                                            "RegIndex", "_", "_"}),
         clause_meta("=/=", {"RegIndex", s2d("NONE")})},
        // -->
        clause_meta("value_reg_unsupported", {"EA", "Reg"}));

    // std::cout << "String Map >> : " << std::endl;
    // for (auto p : string_map) {
    //     std::cout << p.first << " " << p.second << std::endl;
    // }
    // std::cout << "String Map <<" << std::endl;

    SCC_INIT(analysis_scc);
    MEMORY_STAT();
    SCC_COMPUTE(analysis_scc);

    // print all relation name
    // std::cout << "Relation Names: " << std::endl;
    // for (auto p : analysis_scc.relation_name_map) {
    //     std::cout << p.first << std::endl;
    // }
    // std::cout << "Relation Names End" << std::endl;
    // print all updated relation
    std::cout << "Updated relations: >> " << std::endl;
    for (auto p : analysis_scc.update_relations) {
        std::cout << p->name << std::endl;
    }

    PRINT_REL_SIZE(analysis_scc, "block_last_instruction");
    PRINT_REL_SIZE(analysis_scc, "jump_table_target");
    PRINT_REL_SIZE(analysis_scc, "block_next");
    PRINT_REL_SIZE(analysis_scc, "cmp_defines");
    PRINT_REL_SIZE(analysis_scc, "compare_and_jump_immediate");
    PRINT_REL_SIZE(analysis_scc, "compare_and_jump_indirect");
    PRINT_REL_SIZE(analysis_scc, "compare_and_jump_register");
    PRINT_REL_SIZE(analysis_scc, "const_value_reg_used");
    PRINT_REL_SIZE(analysis_scc, "reg_def_use_def_used");
    std::cout << "None value str : " << s2d("NONE") << std::endl;
    PRINT_REL_SIZE(analysis_scc, "def_used_for_address");
    PRINT_REL_SIZE(analysis_scc, "flags_and_jump_pair");
    PRINT_REL_SIZE(analysis_scc, "jump_table_element_access");
    PRINT_REL_SIZE(analysis_scc, "jump_table_signed");
    PRINT_REL_SIZE(analysis_scc, "jump_table_start_tmp1");
    PRINT_REL_SIZE(analysis_scc, "jump_table_start_tmp2");
    PRINT_REL_SIZE(analysis_scc, "jump_table_start");
    PRINT_REL_SIZE(analysis_scc, "jump_table_target");
    PRINT_REL_SIZE(analysis_scc, "last_value_reg_limit");
    PRINT_REL_SIZE(analysis_scc, "last_value_reg_limit_tmp0");
    PRINT_REL_SIZE(analysis_scc, "no_value_reg_limit");
    PRINT_REL_SIZE(analysis_scc, "no_value_reg_limit_debug");
    PRINT_REL_SIZE(analysis_scc, "instruction_non_cmp");
    PRINT_REL_SIZE(analysis_scc, "value_reg_unsupported");
    PRINT_REL_SIZE(analysis_scc, "reg_def_use_def_used");
    PRINT_REL_SIZE(analysis_scc, "reg_def_use_def_used_debug");
    PRINT_REL_SIZE(analysis_scc, "reg_def_use_def_used_tmp2");
    PRINT_REL_SIZE(analysis_scc, "reg_def_use_live_var_used_tmp1");
    PRINT_REL_SIZE(analysis_scc, "reg_def_use_return_val_used");
    PRINT_REL_SIZE(analysis_scc, "value_reg_limit");
}

MAIN_ENTRANCE(run)
