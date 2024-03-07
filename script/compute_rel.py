# a python script to conver souffle datalog rule into GDLog acceptible format

import sys

test_rule = \
    ''''''

test_decl = '''
.decl reg_def_use_def_used(EA_def:address,Var:register,EA_used:address,Index_used:operand_index)
.decl def_used_for_address(EA_def:address,Reg:register,Type:symbol)
.decl reg_used_for(EA:address,Reg:register,Type:symbol)
def_used_for_address(EA_def,Reg,Type) :-
    def_used_for_address(EA_used,_,Type),
    reg_def_use_def_used(EA_def,Reg,EA_used,_).
'''

output_program_name = "analysis_scc"


def skip_comments(lines):
    return [line for line in lines if not line.startswith("//")]


def skip_type_decls(lines):
    return [line for line in lines if not line.startswith(".type")]


def parse_relation_decl(decl_block):
    decl_block = decl_block.strip()
    decl_block = decl_block.strip(".decl")
    decl_block = decl_block.strip()
    relation_name, arguments = decl_block.split("(")
    arguments = arguments.strip(")")
    arguments = arguments.split(",")
    return relation_name, arguments


def parse_relation_decl_block(decl_block):
    lines = decl_block.split("\n")
    rel_decls = {}
    for line in lines:
        if line.startswith(".decl"):
            relation_name, arguments = parse_relation_decl(line)
            rel_decls[relation_name] = arguments
    return rel_decls


INPUT_FLAG = ".input"
OUTPUT_FLAG = ".output"


def parse_datalog_directive(directive):
    directive = directive.strip()
    if directive.startswith(".input"):
        directive = directive.strip(".input")
        directive = directive.strip()
        return INPUT_FLAG, directive
    elif directive.startswith(".output") or \
            directive.startswith(".printsize"):
        directive = directive.strip(".output")
        directive = directive.strip()
        return OUTPUT_FLAG, directive
    return None, None


def parse_datalog_head(head):
    # def_used_for_address(EA_def,Reg,Type)
    # split the head into relation name and arguments
    head = head.strip()
    head = head.strip(":-")
    head = head.strip()
    relation_name, arguments = head.split("(")
    arguments = arguments.strip(")")
    arguments = arguments.split(",")
    return relation_name, arguments


def parse_body(body):
    # def_used_for_address(EA_used,_,Type)
    # reg_def_use_def_used(EA_def,Reg,EA_used,_)
    # split the body into relation name and arguments
    body = body.strip()
    body = body.strip("),")
    body = body.strip(").")
    relation_name, arguments = body.split("(")
    arguments = arguments.strip(")")
    arguments = arguments.split(",")
    return relation_name, arguments


def parse_binary_join_datalog(dl_rule):
    # split the rule into head and body
    clauses = dl_rule.split("\n")
    head = clauses[0]
    bodys = clauses[1:]
    output_name, output_args = parse_datalog_head(head)
    input_rules = []
    for body in bodys:
        relation_name, arguments = parse_body(body)
        input_rules.append((relation_name, arguments))
    return output_name, output_args, input_rules


def parse_transformed_datalog_file(dl_file):
    with open(dl_file, "r") as f:
        lines = f.readlines()
    lines = skip_comments(lines)
    lines = skip_type_decls(lines)
    declared_out = []
    declared_in = []
    decls = []
    binary_rules = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(".decl"):
            decls.append(parse_relation_decl(line))
            i = i + 1
        elif line.startswith(".input") or \
                line.startswith(".output") or \
                line.startswith(".printsize"):
            flag, val = parse_datalog_directive(line)
            if flag == INPUT_FLAG:
                declared_in.append(val)
            elif flag == OUTPUT_FLAG:
                declared_out.append(val)
            i = i + 1
        else:
            output_name, output_args, input_rules = parse_binary_join_datalog(
                '\n'.join(line[i: i + 3]))
            binary_rules.append((output_name, output_args, input_rules))
            i = i + 3
    return declared_out, declared_in, decls, binary_rules


def compute_join_columns(bodys):
    inner = bodys[0]
    outer = bodys[1]
    join_column_mvs = []
    for i, arg in enumerate(inner[1]):
        if arg in outer[1] and arg != "_":
            join_column_mvs.append(arg)
    return join_column_mvs


def reorder_relation_columns(relation_cols, join_columns):
    canonical_order = [i for i in range(len(relation_cols))]
    indexed_join_columns = []
    for i, col in enumerate(relation_cols):
        if col in join_columns:
            canonical_order.remove(i)
            indexed_join_columns.append(i)
    return indexed_join_columns + canonical_order


def binary_join_to_gdlog(binary_rule, decls, input_rels, output_rels, defined_indexed_rels, defined_tmp_rel, defined_cannonical_rels):
    output_name, output_args, input_rules = binary_rule
    jc = compute_join_columns(input_rules)
    generated_rel_decl_strs = []
    generated_c_decl_strs = []
    new_body_clause_lst = []
    inner = input_rules[0]
    inner_name = inner[0]
    inner_metavars = inner[1]
    inner_reorder_idx = reorder_relation_columns(inner[1], jc)
    inner_reorder_idx_str = '_'.join([str(c) for c in inner_reorder_idx])
    outer = input_rules[1]
    outer_name = outer[0]
    outer_metavars = outer[1]
    outer_reorder_idx = reorder_relation_columns(outer[1], jc)
    outer_reorder_idx_str = '_'.join([str(c) for c in outer_reorder_idx])
    inner_cannonical_order = [i for i in range(len(defined_cannonical_rels[inner_name]))]
    outer_cannonical_order = [i for i in range(len(defined_cannonical_rels[outer_name]))]
    inner_gen_flag = inner_reorder_idx == [i for i in range(len(inner_metavars))]
    inner_gen_flag = inner_gen_flag and ((inner_name, inner_reorder_idx) in defined_indexed_rels)
    outer_gen_flag = outer_reorder_idx == [i for i in range(len(outer_metavars))]
    outer_gen_flag = outer_gen_flag and ((outer_name, outer_reorder_idx) in defined_indexed_rels)
    if inner_gen_flag:
        generated_rel_decl_strs.append(f".decl {inner_name}__{inner_reorder_idx_str}__{len(jc)}({','.join([decls[inner_name][i] for i in inner_reorder_idx])})")
        generated_c_decl_strs.append(f"CREATE_FULL_INDEXED_RELATION({output_program_name}, {inner_name}, {len(inner_reorder_idx)}, {inner_reorder_idx_str}, {len(jc)})")
        defined_indexed_rels.append((inner_name, inner_reorder_idx))
    if outer_gen_flag:
        generated_rel_decl_strs.append(f".decl {outer_name}__{outer_reorder_idx_str}__{len(jc)}({','.join([decls[outer_name][i] for i in outer_reorder_idx])})")
        generated_c_decl_strs.append(f"CREATE_TMP_INDEXED_RELATION({output_program_name}, {outer_name}, {len(outer_reorder_idx)}, {outer_reorder_idx_str}, {len(jc)})")
        defined_indexed_rels.append((outer_name, outer_reorder_idx))
    out_clause_str = f"{output_name}({','.join(output_args)}) :-"
    generated_dl_rules = "\n".join(generated_rel_decl_strs) + "\n" + \
        out_clause_str + "\n" + "    \n".join(new_body_clause_lst)
    inner_rel_str = f"INDEXED_REL({inner_name}, {inner_reorder_idx_str}, {len(jc)})"
    if jc == 1 and (inner_reorder_idx == inner_cannonical_order):
        inner_rel_str = inner_name
    outer_rel_str = f"INDEXED_REL({outer_name}, {outer_reorder_idx_str}, {len(jc)})"
    if jc == 1 and (outer_reorder_idx == outer_cannonical_order):
        outer_rel_str = outer_name
    inner_mv_str = f"{{ {','.join([f'"{c}"' for c in inner_reorder_idx])} }}"
    outer_mv_str = f"{{ {','.join([f'"{c}"' for c in outer_reorder_idx])} }}"
    output_mv_str = f"{{ {','.join([f'"{c}"' for c in output_args])} }}"
    if inner_name in output_rels and outer_name in output_rels:
        generated_c_rules = f'''
        SEMI_NAIVE_BINARY_JOIN({output_program_name},
            {inner_rel_str}, {outer_rel_str},
            {inner_mv_str}, {outer_mv_str}, {output_mv_str}
            JOINEND))'''
    elif inner_name in input_rels and outer_name not in output_rels:
        generated_c_rules = f'''
        BINARY_JOIN({output_program_name},
            {inner_rel_str}, FULL, {outer_rel_str}, DELTA,
            {inner_mv_str}, {outer_mv_str}, {output_mv_str}
            JOINEND))'''
    elif inner_name not in output_rels and outer_name in input_rels:
        generated_c_rules = f'''
        BINARY_JOIN({output_program_name},
            {inner_rel_str}, DELTA, {outer_rel_str}, FULL,
            {inner_mv_str}, {outer_mv_str}, {output_mv_str}
            JOINEND))'''
    return generated_dl_rules, generated_c_rules


def generate_c_decl_canonical(decls):
    '''
    .decl def_used_for_address(EA_def:address,Reg:register,Type:symbol)
    -->
    DECLARE_RELATION_INPUT_OUTPUT(analysis_scc, def_used_for_address, 3, 1)
    '''
    c_decl_strs = []
    for rel_name, args in decls:
        c_decl_strs.append(
            f"DECLARE_RELATION_INPUT_OUTPUT({output_program_name}, {rel_name}, {len(args)}, 1);")
    return "\n".join(c_decl_strs)


def transformed_datalog_to_gdlog(datalog_file):
    declared_out, declared_in, decls, binary_rules = parse_transformed_datalog_file(
        datalog_file)
    relation_name_args_map = {}
    for rel_name, args in decls:
        relation_name_args_map[rel_name] = args


# if __name__ == "__main__":
#     main()
