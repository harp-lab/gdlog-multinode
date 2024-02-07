// AST for vanila datalog

#pragma once

#include <string>
#include <variant>
#include <vector>
#include <cstdint>

#include "tokenizer.h"


inline unsigned long string_hash32(const std::string& str) {
    const unsigned long  base = 2166136261u;
    const unsigned long  prime = 16777619u;

    unsigned long  hash = base;
    for (char c: str)
    {
        if ((int)c == 0) continue;
        hash ^= (int)c;
        hash *= prime;
    }
    return hash;
}

inline unsigned long long string_hash(const std::string& str) {
    const unsigned long long base = 14695981039346656037ULL;
    const unsigned long long prime = 1099511628211ULL;
    const unsigned long long c46 = 35184372088832ULL;

    unsigned long long hash = base;
    for (char c: str)
    {
        if ((uint64_t)c == 0) continue;
        hash ^= (uint64_t)c;
        hash *= prime;
    }
    return hash % c46;
}

namespace datalog {
// Forward declaration of visitor class
class DatalogASTVisitor;

// AST for datalog

// enum of all AST node types
enum class DatalogASTNodeType {
    DATALOG_AST_NODE_LIST,
    COLUMN_DEFINITION,
    RELATION_DEFINITION,
    HORN_CLAUSE,
    META_VARIABLE,
    CONSTANT,
    ARITHMETIC_EXPRESSION,
    ARITHMETIC_CLAUSE,
    CONSTRAINT,
    RELATION_CLAUSE,
    DECLARATION_LIST,
    STRATUM,
    DATALOG_PROGRAM
};

class DatalogASTNode {
  public:
    virtual ~DatalogASTNode() = default;
    virtual void accept(DatalogASTVisitor &visitor) = 0;
    DatalogASTNodeType type;
};

class DatalogASTNodeList : public DatalogASTNode {
  public:
    DatalogASTNodeList() = default;
    void add(DatalogASTNode *node);
    void accept(DatalogASTVisitor &visitor) override;

    DatalogASTNode *at(int index) { return nodes[index]; };

    std::vector<DatalogASTNode *> nodes;
};

// enum for column type
enum class ColumnType { INT, STRING };

// ASTNode for attribute definitions
class ColumnDefinition : public DatalogASTNode {
  public:
    ColumnDefinition(const std::string &name, ColumnType type);
    void accept(DatalogASTVisitor &visitor) override;

    std::string name;
    ColumnType col_type;
};

// ASTNode for k-arity relation definitions
// relation ::= (relation_name column_definition_list )
class RelationDefinition : public DatalogASTNode {
  public:
    RelationDefinition(const std::string &name, DatalogASTNodeList *columns);
    void accept(DatalogASTVisitor &visitor) override;

    std::string name;
    DatalogASTNodeList *columns;
};

// ASTNode for horn clause
// HornClause ::= head :- body ... .
class HornClause : public DatalogASTNode {
  public:
    HornClause(DatalogASTNode *head, DatalogASTNodeList *body);
    void accept(DatalogASTVisitor &visitor) override;

    DatalogASTNode *head;
    DatalogASTNodeList *body;
};

// meta varible for horn clause
// MetaVariable ::= var
class MetaVariable : public DatalogASTNode {
  public:
    MetaVariable(const std::string &name);
    void accept(DatalogASTVisitor &visitor) override;

    std::string name;
};

enum class ConstantType { INT, STRING };

// ASTNode for constant
// Constant ::= int | string
class Constant : public DatalogASTNode {
  public:
    Constant(const std::string &value);
    Constant(int value);
    void accept(DatalogASTVisitor &visitor) override;

    std::variant<int, std::string> value;
    ConstantType const_type;
};

// enum for comparison operator
enum class ComparisonOperator { EQ, NE, LT, LE, GT, GE };

// arithmetic operator
enum class ArithmeticOperator { ADD, SUB, MUL, DIV, MOD };

// arithmetic expression
// arithmetic_expression ::= (arithmetic_operator arithmetic_expression
// arithmetic_expression)
//                         | (arithmetic_operator arithmetic_expression
//                         constant) | (arithmetic_operator constant
//                         arithmetic_expression) | (arithmetic_operator
//                         constant constant)
class ArithmeticExpression : public DatalogASTNode {
  public:
    ArithmeticExpression(ArithmeticOperator op, DatalogASTNode *left, DatalogASTNode *right);
    void accept(DatalogASTVisitor &visitor) override;

    ArithmeticOperator op;
    DatalogASTNode *left;
    DatalogASTNode *right;
};

// arithmetic clause ::= (let metavar arithmetic_expression)
class ArithmeticClause : public DatalogASTNode {
  public:
    ArithmeticClause(MetaVariable *var, ArithmeticExpression *expr);
    void accept(DatalogASTVisitor &visitor) override;

    MetaVariable *var;
    ArithmeticExpression *expr;
};

// ASTNode for constriant with comparison
  // Constraint ::= (comparison_operator metavar (arithmetic_expression | constant | metavar))
class Constraint : public DatalogASTNode {
  public:
    Constraint(ComparisonOperator op, DatalogASTNode *left, DatalogASTNode *right);
    void accept(DatalogASTVisitor &visitor) override;

    ComparisonOperator op;
    DatalogASTNode *left;
    DatalogASTNode *right;
};

// relation clause
// RelationClause ::= relation_name ( variable_list )
class RelationClause : public DatalogASTNode {
  public:
    RelationClause(const std::string &name, DatalogASTNodeList *variables);
    void accept(DatalogASTVisitor &visitor) override;

    std::string name;
    DatalogASTNodeList *variables;
};

//  
class DeclarationList : public DatalogASTNode {
  public:
    DeclarationList() = default;
    void add(DatalogASTNode *node);
    void accept(DatalogASTVisitor &visitor) override;

    std::vector<DatalogASTNode *> nodes;
};

class HornClausesList : public DatalogASTNode {
  public:
    HornClausesList() = default;
    void add(DatalogASTNode *node);
    void accept(DatalogASTVisitor &visitor) override;

    std::vector<DatalogASTNode *> nodes;
};

// stratum ::= (stratum name horn_clauses_list)
class Stratum : public DatalogASTNode {
  public:
    Stratum(const std::string &name, DatalogASTNodeList *horn_clauses);
    void accept(DatalogASTVisitor &visitor) override;

    std::string name;
    DatalogASTNodeList *horn_clauses;
};

// DatalogProgram ::= (declaration_list stratum_list)
class DatalogProgram : public DatalogASTNode {
  public:
    DatalogProgram(DatalogASTNodeList *relations, DatalogASTNodeList *stratums);
    void accept(DatalogASTVisitor &visitor) override;

    DatalogASTNodeList *relations;
    DatalogASTNodeList *stratums;
};


// Visitor class for AST traversal
class DatalogASTVisitor {
  public:
    virtual ~DatalogASTVisitor() = default;
    virtual void visit(DatalogASTNodeList &node) = 0;
    virtual void visit(ColumnDefinition &node) = 0;
    virtual void visit(RelationDefinition &node) = 0;
    virtual void visit(HornClause &node) = 0;
    virtual void visit(MetaVariable &node) = 0;
    virtual void visit(Constant &node) = 0;
    virtual void visit(ArithmeticExpression &node) = 0;
    virtual void visit(ArithmeticClause &node) = 0;
    virtual void visit(Constraint &node) = 0;
    virtual void visit(RelationClause &node) = 0;
    virtual void visit(Stratum &node) = 0;
    virtual void visit(DatalogProgram &node) = 0;
};

// parser class
class Parser {
  public:
    Parser(const std::string &input);
    DatalogASTNode *parse();

  private:
    DatalogASTNode *parse_relation_definition();
    DatalogASTNode *parse_column_definition();
    DatalogASTNode *parse_horn_clause();
    DatalogASTNode *parse_meta_variable();
    DatalogASTNode *parse_constant();
    DatalogASTNode *parse_arithmetic_expression();
    DatalogASTNode *parse_arithmetic_clause();
    DatalogASTNode *parse_constraint();
    DatalogASTNode *parse_stratum();
    DatalogASTNode *parse_relation_clause();
    DatalogASTNode *parse_body_clause();
    DatalogASTNode *parse_constraint_right();
    DatalogASTNode *parse_arithmetic_arg();
    DatalogASTNodeList *parse_relation_definition_list();
    DatalogASTNodeList *parse_column_definition_list();
    DatalogASTNodeList *parse_horn_clause_list();
    DatalogASTNodeList *parse_body_clause_list();
    DatalogASTNodeList *parse_variable_list();
    DatalogASTNodeList *parse_constant_list();
    DatalogASTNodeList *parse_arithmetic_expression_list();
    DatalogASTNodeList *parse_constraint_list();
    DatalogASTNodeList *parse_declaration_list();
    DatalogASTNodeList *parse_stratum_list();
    DatalogASTNode *parse_program();

    Tokenizer tokenizer;
};

// AST visitor for printing
class PrintVisitor : public DatalogASTVisitor {
  public:
    void visit(DatalogASTNodeList &node) override;
    void visit(ColumnDefinition &node) override;
    void visit(RelationDefinition &node) override;
    void visit(HornClause &node) override;
    void visit(MetaVariable &node) override;
    void visit(Constant &node) override;
    void visit(ArithmeticExpression &node) override;
    void visit(ArithmeticClause &node) override;
    void visit(Constraint &node) override;
    void visit(RelationClause &node) override;
    void visit(Stratum &node) override;
    void visit(DatalogProgram &node) override;

  private:
    int indent = 0;
};

} // namespace datalog
