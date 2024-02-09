// middle level IR for datalog
// - divide queries into several strongly connected components (not implemented)
// - populate all projection version of relation (index join column)
// - divide all relation into 3 parts: delta, full, new for semi-naive
// evaluation
// - for all arithmetic operation, generate corresponding c++ callback function

#pragma once

// #include "./tuple.cuh"
#include "ast.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace mir {
class MirVisitor;

enum MirNodeType {
    MIR_PROGRAM,
    MIR_RELATION,
    MIR_INDEX,
    MIR_SCC,
    MIR_RULE,
    MIR_COLUMN_META_VAR,
    MIR_COMPARISON,
    MIR_FILTER,
    MIR_PROJECT,
    MIR_ARITHMETIC,
    MIR_GENERATOR,
    MIR_NODE_LIST,
};

class MirNode {
  public:
    MirNodeType type;
    virtual ~MirNode() = default;
    virtual void accept(MirVisitor &visitor) = 0;

    // print the node
    virtual void print() = 0;
    friend std::ostream &operator<<(std::ostream &os, MirNode &node) {
        node.print();
        return os;
    }
};

class MirNodeList : public MirNode {
  public:
    MirNodeList() { type = MIR_NODE_LIST; }
    void add(MirNode *node);
    void accept(MirVisitor &visitor) override;
    void remove(int index) { nodes.erase(nodes.begin() + index); }

    // print
    void print() override {
        for (auto node : nodes) {
            node->print();
        }
    };

    MirNode *at(int index) { return nodes[index]; };

    std::vector<MirNode *> nodes;
};

// MIR Program
class MirProgram : public MirNode {
  public:
    MirProgram() { type = MIR_PROGRAM; }
    void accept(MirVisitor &visitor) override;
    void print() override;
    std::vector<MirNode *> relations;
    std::vector<MirNode *> sccs;
};

// MIR relation declaration
// (relation name arity static_flag io_flag
//           (indices ...))
class MirRelation : public MirNode {
  public:
    MirRelation() { type = MIR_RELATION; }
    void accept(MirVisitor &visitor) override;
    void print() override;
    std::string name;
    int arity;
    bool is_static;
    bool is_input;
    std::vector<MirNode *> indices;
};

// e.g. (1 2 0)
class MirIndex : public MirNode {
  public:
    MirIndex() { type = MIR_INDEX; }
    MirIndex(std::vector<int> &col_pos) {
        type = MIR_INDEX;
        this->col_pos = col_pos;
    }
    void accept(MirVisitor &visitor) override;
    void print() override;
    std::vector<int> col_pos;
};

// MIR scc
class MirScc : public MirNode {
  public:
    MirScc() { type = MIR_SCC; }

    void accept(MirVisitor &visitor) override;
    void print() override;

    std::string name;
    std::vector<MirNode *> *tmp_relations;
    std::vector<MirNode *> updated_relations;
    std::vector<MirNode *> input_relations;
    std::vector<MirNode *> static_ra_ops;
    std::vector<MirNode *> dynamic_ra_ops;

    // for tmp use
    // std::vector<MirNode *> _ra_ops;
};

enum class MirRelationVersion { DELTA, FULL, NEW };

std::ostream &operator<<(std::ostream &os, const MirRelationVersion &version) {
    switch (version) {
    case MirRelationVersion::DELTA:
        os << "DELTA";
        break;
    case MirRelationVersion::FULL:
        os << "FULL";
        break;
    case MirRelationVersion::NEW:
        os << "NEW";
        break;
    default:
        os << "UNKNOWN";
        break;
    }
    return os;
}

// MIR rule
// a rule is a set of input rule, a output relation
// and a set of static RA operation
class MirRule : public MirNode {
  public:
    using input_streams_t = std::map<MirRelation *, MirRelationVersion>;

    MirRule() { type = MIR_RULE; };

    void accept(MirVisitor &visitor) override;
    void print() override;

    MirRelation *output;
    // relation get updated in this rule
    std::map<MirRelation *, MirRelationVersion> stream_relations;
    std::vector<MirNode *> static_relations;
    std::vector<MirNode *> input_relations;

    std::vector<MirNode *> ra_ops;
};

class MirColumnMetaVar : public MirNode {
  public:
    MirColumnMetaVar() { type = MIR_COLUMN_META_VAR; }
    MirColumnMetaVar(std::string name) {
        type = MIR_COLUMN_META_VAR;
        this->name = name;
    }
    void accept(MirVisitor &visitor) override;
    void print() override;
    std::string name;
    int pos;

    // comparator for meta var, just compare the name

    bool operator==(const MirColumnMetaVar &other) const {
        return name == other.name;
    }
};

// left and right are meta var
class MirComparison : public MirNode {
  public:
    MirComparison() { type = MIR_COMPARISON; }
    MirComparison(MirNode *lhs, MirNode *rhs, std::string op) {
        type = MIR_COMPARISON;
        this->lhs = lhs;
        this->rhs = rhs;
        this->op = op;
    }
    void accept(MirVisitor &visitor) override;
    void print() override;

    MirNode *lhs;
    MirNode *rhs;
    std::string op;
};

class MirFilter : public MirNode {
  public:
    MirFilter() { type = MIR_FILTER; }
    MirFilter(MirNode *input_relation, MirNode *output_relation,
              MirNodeList *comparisons) {
        type = MIR_FILTER;
        this->input_relation = input_relation;
        this->output_relation = output_relation;
        this->comparisons = comparisons;
    }
    void accept(MirVisitor &visitor) override;
    void print() override;

    MirNode *input_relation;
    MirNode *output_relation;
    MirNodeList *comparisons;
};

class MirProject : public MirNode {
  public:
    MirProject() { type = MIR_PROJECT; }
    MirProject(MirNode *input, MirNode *output, std::vector<int> &columns) {
        type = MIR_PROJECT;
        this->input = input;
        this->output = output;
        this->reorder_columns = columns;
    }
    void accept(MirVisitor &visitor) override;
    void print() override;

    MirNode *input;
    MirNode *output;
    std::vector<int> reorder_columns;
};

// arithmetic operation for column
class MirArithmetic : public MirNode {
  public:
    MirArithmetic() { type = MIR_ARITHMETIC; }
    MirArithmetic(MirNode *lhs, MirNode *rhs, std::string op) {
        type = MIR_ARITHMETIC;
        this->lhs = lhs;
        this->rhs = rhs;
        this->op = op;
    }
    void accept(MirVisitor &visitor) override;
    void print() override;

    MirNode *lhs;
    MirNode *rhs;
    std::string op;
};

// project and generator
class MirGenerator : public MirNode {
  public:
    MirGenerator() { type = MIR_GENERATOR; }
    MirGenerator(MirNode *input, MirNode *output,
                 std::vector<MirNode *> &arith_ops) {
        type = MIR_GENERATOR;
        this->input = input;
        this->output = output;
        this->arith_ops = arith_ops;
    }
    void accept(MirVisitor &visitor) override;
    void print() override;

    MirNode *input;
    MirNode *output;
    std::vector<MirNode *> arith_ops;
};

class MirVisitor {
  public:
    virtual void visit(MirProgram &node) = 0;
    virtual void visit(MirRelation &node) = 0;
    virtual void visit(MirIndex &node) = 0;
    virtual void visit(MirScc &node) = 0;
    virtual void visit(MirRule &node) = 0;
    virtual void visit(MirColumnMetaVar &node) = 0;
    virtual void visit(MirComparison &node) = 0;
    virtual void visit(MirFilter &node) = 0;
    virtual void visit(MirProject &node) = 0;
    virtual void visit(MirArithmetic &node) = 0;
    virtual void visit(MirGenerator &node) = 0;
    virtual void visit(MirNodeList &node) = 0;
};

// a datalog ast visitor to convert datalog into mir
class DatalogToMirVisitor : public datalog::DatalogASTVisitor {
  public:
    DatalogToMirVisitor();

    void visit(datalog::DatalogASTNodeList &node) override;
    void visit(datalog::ColumnDefinition &node) override;
    void visit(datalog::RelationDefinition &node) override;
    void visit(datalog::HornClause &node) override;
    void visit(datalog::MetaVariable &node) override;
    void visit(datalog::Constant &node) override;
    void visit(datalog::ArithmeticExpression &node) override;
    void visit(datalog::Constraint &node) override;
    void visit(datalog::RelationClause &node) override;
    void visit(datalog::Stratum &node) override;
    void visit(datalog::DatalogProgram &node) override;

    MirProgram *get_mir_program() { return program; }

  private:
    MirProgram *program = nullptr;
    MirRelation *current_relation = nullptr;
    MirRule::input_streams_t current_rule_streams;
    std::vector<MirNode *> current_rule_static_relations;
    std::vector<MirNode *> *current_rule_dynamic_relations;
    // tmp for stratum under processing
    MirScc *current_stratum = nullptr;
    // buffer for current node list
    std::vector<MirNode *> current_node_list;
    std::vector<int> current_index;
    MirNode *current_node = nullptr;

    // for clause under processing
    MirRelation *tmp_left_relation = nullptr;
    int current_clause_pos = 0;
    MirRelation *current_output_relation = nullptr;
    std::vector<std::string> current_output_meta_vars;
    std::map<datalog::DatalogASTNode *, std::vector<std::string>>
        current_body_meta_vars;

    // fo metavar under processing
    std::vector<MirNode *> current_columns_list;
    MirColumnMetaVar *current_meta_var = nullptr;
    std::vector<MirColumnMetaVar *> current_meta_var_list;
    std::vector<MirColumnMetaVar *> prev_meta_var_list;
    // std::vector<std::string> current_meta_var_names;

    MirRule *current_rule;

    MirRelation *find_relation_by_name(const std::string &name);
    std::vector<std::string> meta_var_in_clause(datalog::DatalogASTNode *node);
};

// MIR rule
} // namespace mir
