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
class ASTVisitor;

// AST for datalog

class ASTNode {
  public:
    virtual ~ASTNode() = default;
    virtual void accept(ASTVisitor &visitor) = 0;
};

class ASTNodeList : public ASTNode {
  public:
    ASTNodeList() = default;
    void add(ASTNode *node);
    void accept(ASTVisitor &visitor) override;

    ASTNode *at(int index) { return nodes[index]; };

    std::vector<ASTNode *> nodes;
};

// enum for column type
enum class ColumnType { INT, STRING };

// ASTNode for attribute definitions
class ColumnDefinition : public ASTNode {
  public:
    ColumnDefinition(const std::string &name, ColumnType type);
    void accept(ASTVisitor &visitor) override;

    std::string name;
    ColumnType type;
};

// ASTNode for k-arity relation definitions
// relation ::= (relation_name column_definition_list )
class RelationDefinition : public ASTNode {
  public:
    RelationDefinition(const std::string &name, ASTNodeList *columns);
    void accept(ASTVisitor &visitor) override;

    std::string name;
    ASTNodeList *columns;
};

// ASTNode for horn clause
// HornClause ::= head :- body ... .
class HornClause : public ASTNode {
  public:
    HornClause(ASTNode *head, ASTNodeList *body);
    void accept(ASTVisitor &visitor) override;

    ASTNode *head;
    ASTNodeList *body;
};

// meta varible for horn clause
// MetaVariable ::= var
class MetaVariable : public ASTNode {
  public:
    MetaVariable(const std::string &name);
    void accept(ASTVisitor &visitor) override;

    std::string name;
};

enum class ConstantType { INT, STRING };

// ASTNode for constant
// Constant ::= int | string
class Constant : public ASTNode {
  public:
    Constant(const std::string &value);
    Constant(int value);
    void accept(ASTVisitor &visitor) override;

    std::variant<int, std::string> value;
    ConstantType type;
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
class ArithmeticExpression : public ASTNode {
  public:
    ArithmeticExpression(ArithmeticOperator op, ASTNode *left, ASTNode *right);
    void accept(ASTVisitor &visitor) override;

    ArithmeticOperator op;
    ASTNode *left;
    ASTNode *right;
};

// ASTNode for constriant with comparison
// Constraint ::= (comparison_operator arithmetic_expression
// arithmetic_expression)
class Constraint : public ASTNode {
  public:
    Constraint(ComparisonOperator op, ASTNode *left, ASTNode *right);
    void accept(ASTVisitor &visitor) override;

    ComparisonOperator op;
    ASTNode *left;
    ASTNode *right;
};

// relation clause
// RelationClause ::= relation_name ( variable_list )
class RelationClause : public ASTNode {
  public:
    RelationClause(const std::string &name, ASTNodeList *variables);
    void accept(ASTVisitor &visitor) override;

    std::string name;
    ASTNodeList *variables;
};

//  
class DeclarationList : public ASTNode {
  public:
    DeclarationList() = default;
    void add(ASTNode *node);
    void accept(ASTVisitor &visitor) override;

    std::vector<ASTNode *> nodes;
};

class HornClausesList : public ASTNode {
  public:
    HornClausesList() = default;
    void add(ASTNode *node);
    void accept(ASTVisitor &visitor) override;

    std::vector<ASTNode *> nodes;
};

class DatalogProgram : public ASTNode {
  public:
    DatalogProgram(ASTNodeList *relations, ASTNodeList *hclauses);
    void accept(ASTVisitor &visitor) override;

    ASTNodeList *relations;
    ASTNodeList *clauses;
};


// Visitor class for AST traversal
class ASTVisitor {
  public:
    virtual ~ASTVisitor() = default;
    virtual void visit(ASTNodeList &node) = 0;
    virtual void visit(ColumnDefinition &node) = 0;
    virtual void visit(RelationDefinition &node) = 0;
    virtual void visit(HornClause &node) = 0;
    virtual void visit(MetaVariable &node) = 0;
    virtual void visit(Constant &node) = 0;
    virtual void visit(ArithmeticExpression &node) = 0;
    virtual void visit(Constraint &node) = 0;
    virtual void visit(RelationClause &node) = 0;
    virtual void visit(DatalogProgram &node) = 0;
};

// parser class
class Parser {
  public:
    Parser(const std::string &input);
    ASTNode *parse();

  private:
    ASTNode *parse_relation_definition();
    ASTNode *parse_column_definition();
    ASTNode *parse_horn_clause();
    ASTNode *parse_meta_variable();
    ASTNode *parse_constant();
    ASTNode *parse_arithmetic_expression();
    ASTNode *parse_constraint();
    ASTNode *parse_relation_clause();
    ASTNodeList *parse_relation_definition_list();
    ASTNodeList *parse_column_definition_list();
    ASTNodeList *parse_horn_clause_list();
    ASTNodeList *parse_body_clause_list();
    ASTNodeList *parse_variable_list();
    ASTNodeList *parse_constant_list();
    ASTNodeList *parse_arithmetic_expression_list();
    ASTNodeList *parse_constraint_list();
    ASTNodeList *parse_declaration_list();
    ASTNode *parse_program();

    Tokenizer tokenizer;
};

// AST visitor for printing
class PrintVisitor : public ASTVisitor {
  public:
    void visit(ASTNodeList &node) override;
    void visit(ColumnDefinition &node) override;
    void visit(RelationDefinition &node) override;
    void visit(HornClause &node) override;
    void visit(MetaVariable &node) override;
    void visit(Constant &node) override;
    void visit(ArithmeticExpression &node) override;
    void visit(Constraint &node) override;
    void visit(RelationClause &node) override;
    void visit(DatalogProgram &node) override;

  private:
    int indent = 0;
};

} // namespace datalog
