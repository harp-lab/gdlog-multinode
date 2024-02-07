
#include "../include/mir.h"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>

namespace mir {

void MirNodeList::add(MirNode *node) { nodes.push_back(node); }
void MirNodeList::accept(MirVisitor &visitor) {
    visitor.visit(*this);
    for (auto node : nodes) {
        node->accept(visitor);
    }
}

void MirProgram::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirRelation::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirIndex::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirScc::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirRule::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirColumnMetaVar::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirComparison::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirFilter::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirProject::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirArithmetic::accept(MirVisitor &visitor) { visitor.visit(*this); }
void MirGenerator::accept(MirVisitor &visitor) { visitor.visit(*this); }

// printer for Mir
void MirProgram::print() {
    std::cout << "(MirProgram" << std::endl;
    for (auto node : relations) {
        node->print();
    }
    for (auto node : sccs) {
        node->print();
    }
    std::cout << ")" << std::endl;
}

void MirRelation::print() {
    std::cout << "(MirRelation " << name << " " << arity << " " << is_static
              << " " << is_input << " ";
    for (auto index : indices) {
        index->print();
    }
    std::cout << ")" << std::endl;
}

void MirIndex::print() {
    std::cout << "(MirIndex ";
    for (auto col : col_pos) {
        std::cout << col << " ";
    }
    std::cout << ")" << std::endl;
}

void MirScc::print() {
    std::cout << "(MirScc" << std::endl;
    for (auto node : updated_relations) {
        node->print();
    }
    for (auto node : static_ra_ops) {
        node->print();
    }
    for (auto node : dynamic_ra_ops) {
        node->print();
    }
    std::cout << ")" << std::endl;
}

void MirRule::print() {
    std::cout << "(MirRule" << std::endl;
    output->print();
    for (auto &stream : stream_relations) {
        std::cout << stream.first->name << " " << stream.second << std::endl;
    }
    for (auto node : static_relations) {
        node->print();
    }
    std::cout << ")" << std::endl;
}

void MirColumnMetaVar::print() {
    std::cout << "(MirColumnMetaVar " << name << ")" << std::endl;
}

void MirComparison::print() {
    std::cout << "(MirComparison " << op << ")" << std::endl;
    lhs->print();
    rhs->print();
}

void MirFilter::print() {
    std::cout << "(MirFilter" << std::endl;
    input_relation->print();
    output_relation->print();
    comparisons->print();
    std::cout << ")" << std::endl;
}

void MirProject::print() {
    std::cout << "(MirProject" << std::endl;
    input->print();
    output->print();
    for (auto col : reorder_columns) {
        std::cout << col << " ";
    }
    std::cout << ")" << std::endl;
}

void MirArithmetic::print() {
    std::cout << "(MirArithmetic " << op << ")" << std::endl;
    lhs->print();
    rhs->print();
}

void MirGenerator::print() {
    std::cout << "(MirGenerator" << std::endl;
    input->print();
    output->print();
    for (auto col : arith_ops) {
        std::cout << col << " ";
    }
    std::cout << ")" << std::endl;
}

// a datalog ast visitor to convert datalog into mir

MirRelation *
DatalogToMirVisitor::find_relation_by_name(const std::string &name) {
    for (auto node : program->relations) {
        // cast relation to MirRelation
        const auto &relation = static_cast<MirRelation *>(node);
        if (relation->name == name) {
            return relation;
        }
    }
    return nullptr;
}

DatalogToMirVisitor::DatalogToMirVisitor() { program = new MirProgram(); }

// convert datalog program into mir program
void DatalogToMirVisitor::visit(datalog::DatalogProgram &node) {
    program->sccs = new MirNodeList();
    program->relations = new MirNodeList();

    // convert each relation decl into MirRelation
    for (auto &node : node.relations->nodes) {
        node->accept(*this);
        program->relations.push_back(current_relation);
    }

    // convert each stratum into a SCC
    for (auto &node : node.stratums->nodes) {
        // visit each stratum
        node->accept(*this);
        program->sccs.push_back(current_stratum);
    }
}

void DatalogToMirVisitor::visit(datalog::Stratum &node) {
    current_stratum = new MirScc();
    current_stratum->updated_relations = new MirNodeList();
    current_stratum->static_ra_ops = new MirNodeList();
    current_stratum->dynamic_ra_ops = new MirNodeList();
    // visit each horn clause
    for (auto &node : node.horn_clauses->nodes) {
        node->accept(*this);
    }
}

void DatalogToMirVisitor::visit(datalog::DatalogASTNodeList &node) {
    // NOTE: don't use this
    current_node_list = new MirNodeList();
    for (auto &child : node.nodes) {
        child->accept(*this);
    }
}

void DatalogToMirVisitor::visit(datalog::ColumnDefinition &node) {
    // NOTE: not used
    auto col = new MirColumnMetaVar(node.name);
    col->name = node.name;
    current_meta_var = col;
}

void DatalogToMirVisitor::visit(datalog::RelationDefinition &node) {
    auto relation = new MirRelation();
    relation->name = node.name;
    relation->arity = node.columns->nodes.size();
    // name starts with @ is input relation
    relation->is_input = node.name[0] == '@';
    // init to true, change to false when it appears in a rule's head
    relation->is_static = false;
    for (auto &sr_pair : static_relations) {
        auto stratum = sr_pair.first;
        auto &static_relation_list = sr_pair.second;
        if (std::find(static_relation_list.begin(), static_relation_list.end(),
                      node.name) != static_relation_list.end()) {
            relation->is_static = true;
            break;
        }
    }
    // every relation start with a canonical index
    relation->indices = new MirNodeList();
    // a canonical index is a sequence of 0, 1, 2, 3, ... arity - 1
    std::vector<int> canonical_index(relation->arity);
    relation->indices->add(new MirIndex(canonical_index));
    program->relations.push_back(relation);
    // all declared relations has a full version
    current_relation = relation;
}

std::vector<std::string>
DatalogToMirVisitor::meta_var_in_clause(datalog::DatalogASTNode *node) {
    std::vector<std::string> meta_vars;
    // collect meta vars in each body clause
    if (node->type == datalog::DatalogASTNodeType::RELATION_CLAUSE) {
        auto dl_body_clause = static_cast<datalog::RelationClause *>(node);
        for (auto &arg : dl_body_clause->variables->nodes) {
            meta_vars.push_back(static_cast<datalog::MetaVariable *>(arg)->name);
        }
    } else if (node->type == datalog::DatalogASTNodeType::CONSTRAINT) {
        auto dl_body_clause = static_cast<datalog::Constraint *>(node);
        auto left = static_cast<datalog::MetaVariable *>(dl_body_clause->left);
        meta_vars.push_back(left->name);
        auto right = dl_body_clause->right;
        if (right->type == datalog::DatalogASTNodeType::META_VARIABLE) {
            auto mv = static_cast<datalog::MetaVariable *>(right);
            meta_vars.push_back(mv->name);
        }
    } else if (node->type ==
               datalog::DatalogASTNodeType::ARITHMETIC_EXPRESSION) {
        auto dl_body_clause =
            static_cast<datalog::ArithmeticExpression *>(node);
        auto left = static_cast<datalog::MetaVariable *>(dl_body_clause->left);
        // no need to collect meta var in left, as it must be in the output
        auto right = dl_body_clause->right;
        if (right->type == datalog::DatalogASTNodeType::META_VARIABLE) {
            auto mv = static_cast<datalog::MetaVariable *>(right);
            meta_vars.push_back(mv->name);
        }
    } else if (node->type == datalog::DatalogASTNodeType::META_VARIABLE) {
        auto dl_body_clause = static_cast<datalog::MetaVariable *>(node);
        // current_body_meta_vars[node].push_back(dl_body_clause->name);
        current_body_meta_vars.push_back(std::make_tuple(node, meta_vars));
    } else if (node->type == datalog::DatalogASTNodeType::CONSTANT) {
        // pass
    }
    return meta_vars;
}

void DatalogToMirVisitor::visit(datalog::HornClause &node) {
    tmp_left_relation = nullptr;
    current_output_relation = nullptr;
    current_clause_pos = 0;
    current_rule = new MirRule();
    // convert the output relation
    node.head->accept(*this);
    current_rule->output = current_output_relation;

    auto total_input_clause_count = node.body->nodes.size();
    // collect meta vars in each body clause
    for (int i = 0; i < total_input_clause_count; i++) {
        auto body_clause = node.body->nodes[i];
        current_body_meta_vars.push_back(
            std::make_tuple(body_clause, meta_var_in_clause(body_clause)));
    }

    // compile each body clause
    for (auto &body_clause : node.body->nodes) {
        current_clause_pos++;
        body_clause->accept(*this);
    }
}

void DatalogToMirVisitor::visit(datalog::RelationClause &node) {
    // prev_meta_var_list.swap(current_meta_var_list);
    // current_meta_var_list.clear();
    // int i = 0;
    // for (auto &arg : node.variables->nodes) {
    //     arg->accept(*this);
    //     current_meta_var->pos = i;
    //     current_meta_var_list.push_back(current_meta_var);
    // }
    if (current_output_relation == nullptr) {
        // visiting the output relation of a horn clause
        current_output_relation = find_relation_by_name(node.name);
        for (auto mv : current_meta_var_list) {
            current_output_meta_vars.push_back(mv->name);
        }
        // mark the output relation as non-static
        // current_output_relation->is_static = false;
        // mark it as updated in the current stratum
        current_stratum->updated_relations.push_back(current_output_relation);
    }

    auto total_input_clause_count = current_body_meta_vars.size();
    // find cur_meta_var in list current_body_meta_vars
    bool found_input_clause = false;
    std::vector<std::string> cur_meta_var;
    for (auto &input_clause : current_body_meta_vars) {
        if (std::get<0>(input_clause) == &node) {
            found_input_clause = true;
            cur_meta_var = std::get<1>(input_clause);
            break;
        }
    }
    if (tmp_left_relation == nullptr && total_input_clause_count == 1) {
        // a body clause, and this is the only one, interepret as a project
        // operation
        auto project = new MirProject();
        project->output = current_output_relation;
        tmp_left_relation = find_relation_by_name(node.name);
        if (tmp_left_relation == nullptr) {
            std::runtime_error("relation " + node.name + " not found");
        }

        project->input = tmp_left_relation;
        // match the meta var name in output relation with the meta var name in
        // put the pos of mv in input relation to the reorder_columns
        for (auto mv : current_output_meta_vars) {
            auto pos = std::find(cur_meta_var.begin(), cur_meta_var.end(), mv);
            if (pos == cur_meta_var.end()) {
                std::runtime_error("meta var " + mv + " not found in relation " + node.name);
            }
            project->reorder_columns.push_back(
                std::distance(cur_meta_var.begin(), pos));
            current_rule->ra_ops.push_back(project);
        }
    } else if (tmp_left_relation == nullptr && total_input_clause_count > 1) {
        // the head of a sequence of body clauses
        // mark it as current left
        tmp_left_relation = find_relation_by_name(node.name);
        if (tmp_left_relation == nullptr) {
            std::runtime_error("relation " + node.name + " not found");
        }

        // mark it as input relation in current rule
        current_rule->input_relations.push_back(tmp_left_relation);
        std::vector<int> indexed_column_pos_list;
        // compute the index by refering to set interset will all meta var used
        // later
        for (size_t i = 0; i < current_body_meta_vars.size(); i++) {
            auto clasue_mv_pair = current_body_meta_vars[i];
            auto clause = std::get<0>(clasue_mv_pair);
            auto clause_meta_vars = std::get<1>(clasue_mv_pair);
            if (clause == &node) {
                for (size_t j = i + 1; j < current_body_meta_vars.size(); j++) {
                    auto later_clasue_mv_pair = current_body_meta_vars[j];
                    auto later_clause = std::get<0>(later_clasue_mv_pair);
                    auto later_clause_meta_vars = std::get<1>(later_clasue_mv_pair);
                    for (auto mv : clause_meta_vars) {
                        auto pos = std::find(later_clause_meta_vars.begin(),
                                             later_clause_meta_vars.end(), mv);
                        if (pos != later_clause_meta_vars.end()) {
                            // column used in join/ RA operation later
                            // mark it as an indexed column
                            indexed_column_pos_list.push_back(i);
                        }
                    }
                }  
            }
        }

    } else if (tmp_left_relation != nullptr &&
               current_clause_pos != total_input_clause_count) {
        // a body clause in the middle of a sequence of body clauses
        // compile to a join with current left relation
    }
}

void DatalogToMirVisitor::visit(datalog::MetaVariable &node) {
    auto col = new MirColumnMetaVar(node.name);
    current_meta_var = col;
    // current_meta_var_list.push_back(col);
}


void DatalogRelationVisitor::visit(datalog::RelationClause &node) {
    if (processing_input_relation) {
        // visiting the input relation of a horn clause
        // check if this rule has already been mark as static or dynamic
        bool static_huh = std::find(static_relations[current_stratum].begin(),
                      static_relations[current_stratum].end(), node.name) !=
            static_relations[current_stratum].end();
        bool dynamic_huh = std::find(dynamic_relations[current_stratum].begin(),
                      dynamic_relations[current_stratum].end(), node.name) !=
            dynamic_relations[current_stratum].end();
        if (dynamic_huh) {
            return;
        } else if (static_huh) {
            return;
        } else {
            static_relations[current_stratum].push_back(node.name);
        }
    } else {
        // visiting the output relation of a horn clause
        auto place_in_static = std::find(static_relations[current_stratum].begin(),
                      static_relations[current_stratum].end(), node.name);
        auto place_in_dynamic = std::find(dynamic_relations[current_stratum].begin(),
                      dynamic_relations[current_stratum].end(), node.name);
        if (place_in_dynamic != dynamic_relations[current_stratum].end()) {
            return;
        } else if (place_in_static != static_relations[current_stratum].end()) {
            static_relations[current_stratum].erase(place_in_static);
            dynamic_relations[current_stratum].push_back(node.name);
        } else {
            dynamic_relations[current_stratum].push_back(node.name);
        }
    }
}

void DatalogRelationVisitor::visit(datalog::HornClause &node) {
    processing_input_relation = true;
    for (auto &body_clause : node.body->nodes) {
        body_clause->accept(*this);
    }
    processing_input_relation = false;
    node.head->accept(*this);
}

void DatalogRelationVisitor::visit(datalog::DatalogProgram &node) {
    for (auto &stratum : node.stratums->nodes) {
        current_stratum = static_cast<datalog::Stratum *>(stratum)->name;
        static_relations[current_stratum] = std::vector<std::string>();
        dynamic_relations[current_stratum] = std::vector<std::string>();
        stratum->accept(*this);
    }
}

} // namespace mir
