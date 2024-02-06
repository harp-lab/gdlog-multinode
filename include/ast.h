
#pragma once

#include <vector>
#include <string>


namespace datalog
{
    

// tokenizer for datalog

// define end term
enum class TokenType {
    END,
    LPAREN,
    RPAREN,
    COMMA,
    DOT,
    COLON,
    SEMICOLON,
    QUESTION,
    IDENTIFIER,
    INT,
    STRING,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    ADD,
    SUB,
    MUL,
    DIV,
    MOD
};


// AST for datalog

class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual void print() const = 0;
    virtual void generate_code() const = 0;
};

class ASTNodeList : public ASTNode {
public:
    ASTNodeList() = default;
    void add(ASTNode* node);
    void print() const override;
    void generate_code() const override;
private:
    std::vector<ASTNode*> nodes;
};


// enum for column type
enum class ColumnType {
    INT,
    STRING
};

// ASTNode for attribute definitions
class ColumnDefinition : public ASTNode {
public:
    ColumnDefinition(const std::string& name, ColumnType type);
    void print() const override;
    void generate_code() const override;
private:
    std::string name;
    ColumnType type;
};

// ASTNode for k-arity relation definitions
// relation ::= relation_name ( column_definition_list )
class RelationDefinition : public ASTNode {
public:
    RelationDefinition(const std::string& name, ASTNodeList* columns);
    void print() const override;
    void generate_code() const override;
private:
    std::string name;
    ASTNodeList* columns;
};

// ASTNode for horn clause
// HornClause ::= head :- body ... .
class HornClause : public ASTNode {
public:
    HornClause(ASTNode* head, ASTNodeList* body);
    void print() const override;
    void generate_code() const override;
private:
    ASTNode* head;
    ASTNodeList* body;
};

// meta varible for horn clause
// MetaVariable ::= ?var
class MetaVariable : public ASTNode {
public:
    MetaVariable(const std::string& name);
    void print() const override;
    void generate_code() const override;
private:
    std::string name;
};

// enum for comparison operator
enum class ComparisonOperator {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE
};

// arithmetic operator
enum class ArithmeticOperator {
    ADD,
    SUB,
    MUL,
    DIV,
    MOD
};

// arithmetic expression
// arithmetic_expression ::= (arithmetic_operator arithmetic_expression arithmetic_expression)
//                         | (arithmetic_operator arithmetic_expression constant)
//                         | (arithmetic_operator constant arithmetic_expression)
//                         | (arithmetic_operator constant constant)
class ArithmeticExpression : public ASTNode {
public:
    ArithmeticExpression(ArithmeticOperator op, ASTNode* left, ASTNode* right);
    void print() const override;
    void generate_code() const override;
private:
    ArithmeticOperator op;
    ASTNode* left;
    ASTNode* right;
};

// ASTNode for constriant with comparison
// Constraint ::= (comparison_operator arithmetic_expression arithmetic_expression)
class Constraint : public ASTNode {
public:
    Constraint(ComparisonOperator op, ASTNode* left, ASTNode* right);
    void print() const override;
    void generate_code() const override;
private:
    ComparisonOperator op;
    ASTNode* left;
    ASTNode* right;
};

// relation clause
// RelationClause ::= relation_name ( variable_list )
class RelationClause : public ASTNode {
public:
    RelationClause(const std::string& name, ASTNodeList* variables);
    void print() const override;
    void generate_code() const override;
private:    
    std::string name;
    ASTNodeList* variables;
};

// body clause can be either a clause or a constraint
// head clause can only be a clause



} // namespace datalog
