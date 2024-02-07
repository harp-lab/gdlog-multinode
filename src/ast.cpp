
#include "../include/ast.h"
#include "../include/dynamic_dispatch.h"

#include <iostream>
#include <string>
#include <vector>
#include <format>

namespace datalog {
// write implementation based on above commented definitions
// tokenizer has been implemented in tokenizer.cpp

// ASTNodeList
void ASTNodeList::add(ASTNode *node) { nodes.push_back(node); }

void ASTNodeList::accept(ASTVisitor &visitor) { visitor.visit(*this); }

// ColumnDefinition
ColumnDefinition::ColumnDefinition(const std::string &name, ColumnType type)
    : name(name), type(type) {}
void ColumnDefinition::accept(ASTVisitor &visitor) { visitor.visit(*this); }

// RelationDefinition
RelationDefinition::RelationDefinition(const std::string &name,
                                       ASTNodeList *columns)
    : name(name), columns(columns) {}
void RelationDefinition::accept(ASTVisitor &visitor) { visitor.visit(*this); }

// HornClause
HornClause::HornClause(ASTNode *head, ASTNodeList *body)
    : head(head), body(body) {}
void HornClause::accept(ASTVisitor &visitor) { visitor.visit(*this); }

// MetaVariable
MetaVariable::MetaVariable(const std::string &name) : name(name) {}
void MetaVariable::accept(ASTVisitor &visitor) { visitor.visit(*this); }

// ArithmeticExpression
ArithmeticExpression::ArithmeticExpression(ArithmeticOperator op, ASTNode *left,
                                           ASTNode *right)
    : op(op), left(left), right(right) {}
void ArithmeticExpression::accept(ASTVisitor &visitor) { visitor.visit(*this); }

// Constant
Constant::Constant(int value) : value(value), type(ConstantType::INT) {}
Constant::Constant(const std::string &value) : value(value), type(ConstantType::STRING) {}
void Constant::accept(ASTVisitor &visitor) { visitor.visit(*this); }

// Constraint
Constraint::Constraint(ComparisonOperator op, ASTNode *left, ASTNode *right)
    : op(op), left(left), right(right) {}
void Constraint::accept(ASTVisitor &visitor) { visitor.visit(*this); }

// RelationClause
RelationClause::RelationClause(const std::string &name, ASTNodeList *variables)
    : name(name), variables(variables) {}
void RelationClause::accept(ASTVisitor &visitor) { visitor.visit(*this); }

void DeclarationList::add(ASTNode *node) { nodes.push_back(node); }

void HornClausesList::add(ASTNode *node) { nodes.push_back(node); }

DatalogProgram::DatalogProgram(ASTNodeList *relations, ASTNodeList *clauses)
    : relations(relations), clauses(clauses) {}
void DatalogProgram::accept(ASTVisitor &visitor) { visitor.visit(*this); }

// Parser
Parser::Parser(const std::string &input) : tokenizer(input) {}

ASTNode *Parser::parse() {
    return parse_program();    
}

ASTNode *Parser::parse_relation_definition() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    tokenizer.expect(Token{TokenType::RELATION, "relation"});
    auto name_token = tokenizer.next_token();
    auto name = name_token.str;
    auto columns = parse_column_definition_list();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new RelationDefinition(name, columns);
}

ASTNode *Parser::parse_column_definition() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto name_token = tokenizer.next_token();
    auto name = name_token.str;
    auto type_token = tokenizer.next_token();
    auto type = type_token.type == TokenType::INT ? ColumnType::INT
                                                  : ColumnType::STRING;
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new ColumnDefinition(name, type);
}

ASTNode *Parser::parse_horn_clause() {
    tokenizer.expect(Token{TokenType::LPRACKET, "["});
    auto head = parse_relation_clause();
    tokenizer.expect(Token{TokenType::LARROW, "<--"});
    auto body = parse_body_clause_list();
    tokenizer.expect(Token{TokenType::RPRACKET, "]"});
    return new HornClause(head, body);
}

ASTNodeList *Parser::parse_body_clause_list() {
    auto list = new ASTNodeList();
    while (tokenizer.peak().type != TokenType::RPRACKET) {
        list->add(parse_relation_clause());
    }
    return list;
}

ASTNode *Parser::parse_meta_variable() {
    auto name_token = tokenizer.next_token();
    return new MetaVariable(name_token.str);
}

ASTNode *Parser::parse_constant() {
    auto token = tokenizer.next_token();
    if (token.type == TokenType::INT) {
        return new Constant(std::stoi(token.str));
    } else {
        return new Constant(token.str);
    }
}

ASTNode *Parser::parse_arithmetic_expression() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto op_token = tokenizer.next_token();
    auto op = op_token.str;
    auto left = parse_arithmetic_expression();
    auto right = parse_arithmetic_expression();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new ArithmeticExpression(op == "+"   ? ArithmeticOperator::ADD
                                    : op == "-" ? ArithmeticOperator::SUB
                                    : op == "*" ? ArithmeticOperator::MUL
                                    : op == "/" ? ArithmeticOperator::DIV
                                                : ArithmeticOperator::MOD,
                                    left, right);
}

ASTNode *Parser::parse_constraint() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto op_token = tokenizer.next_token();
    auto op = op_token.str;
    auto left = parse_arithmetic_expression();
    auto right = parse_arithmetic_expression();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new Constraint(op == "=="   ? ComparisonOperator::EQ
                          : op == "!=" ? ComparisonOperator::NE
                          : op == "<"  ? ComparisonOperator::LT
                          : op == "<=" ? ComparisonOperator::LE
                          : op == ">"  ? ComparisonOperator::GT
                                       : ComparisonOperator::GE,
                          left, right);
}

ASTNode *Parser::parse_relation_clause() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto token = tokenizer.next_token();
    auto name = token.str;
    auto variables = parse_variable_list();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new RelationClause(name, variables);
}

ASTNodeList *Parser::parse_relation_definition_list() {
    auto list = new ASTNodeList();
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_relation_definition());
    }
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return list;
}

ASTNodeList *Parser::parse_column_definition_list() {
    auto list = new ASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_column_definition());
    }
    return list;
}

ASTNodeList *Parser::parse_horn_clause_list() {
    auto list = new ASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_horn_clause());
    }
    return list;
}

ASTNodeList *Parser::parse_variable_list() {
    auto list = new ASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_meta_variable());
    }
    return list;
}

ASTNodeList *Parser::parse_constant_list() {
    auto list = new ASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_constant());
    }
    return list;
}

ASTNodeList *Parser::parse_arithmetic_expression_list() {
    auto list = new ASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_arithmetic_expression());
    }
    return list;
}

ASTNodeList *Parser::parse_constraint_list() {
    auto list = new ASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_constraint());
    }
    return list;
}

ASTNodeList *Parser::parse_declaration_list() {
    auto list = new ASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_relation_definition());
    }
    return list;
}

// Program ::= ( (RelationDefinitionList) HornClauseList )
ASTNode* Parser::parse_program() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto relations = parse_relation_definition_list();
    auto clauses = parse_horn_clause_list();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new DatalogProgram(relations, clauses);
}

// PrintVisitor
void PrintVisitor::visit(ASTNodeList &node) {
    for (auto n : node.nodes) {
        n->accept(*this);
    }
}

void PrintVisitor::visit(ColumnDefinition &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "ColumnDefinition(" << node.name << ", "
              << (node.type == ColumnType::INT ? "int" : "string") << ")\n";
}

void PrintVisitor::visit(RelationDefinition &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "RelationDefinition(" << node.name << ",\n";
    indent++;
    node.columns->accept(*this);
    indent--;
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << ")\n";
}

void PrintVisitor::visit(HornClause &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "HornClause(\n";
    indent++;
    node.head->accept(*this);
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << ":-\n";
    node.body->accept(*this);
    indent--;
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << ")\n";
}

void PrintVisitor::visit(MetaVariable &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "MetaVariable(" << node.name << ")\n";
}

void PrintVisitor::visit(Constant &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    //
    std::cout << "Constant(";
    // match variable type
    if (node.type == ConstantType::INT) {
        std::cout << std::get<int>(node.value);
    } else {
        std::cout << "\"" << std::get<std::string>(node.value) << "\"";
    }
}

void PrintVisitor::visit(ArithmeticExpression &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "ArithmeticExpression(";
    switch (node.op) {
    case ArithmeticOperator::ADD:
        std::cout << "+";
        break;
    case ArithmeticOperator::SUB:
        std::cout << "-";
        break;
    case ArithmeticOperator::MUL:
        std::cout << "*";
        break;
    case ArithmeticOperator::DIV:
        std::cout << "/";
        break;
    case ArithmeticOperator::MOD:
        std::cout << "%";
        break;
    }
    std::cout << ", ";
    node.left->accept(*this);
    std::cout << ", ";
    node.right->accept(*this);
    std::cout << ")\n";
}

void PrintVisitor::visit(Constraint &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "Constraint(";
    switch (node.op) {
    case ComparisonOperator::EQ:
        std::cout << "==";
        break;
    case ComparisonOperator::NE:
        std::cout << "!=";
        break;
    case ComparisonOperator::LT:
        std::cout << "<";
        break;
    case ComparisonOperator::LE:
        std::cout << "<=";
        break;
    case ComparisonOperator::GT:
        std::cout << ">";
        break;
    case ComparisonOperator::GE:
        std::cout << ">=";
        break;
    }
    std::cout << ", ";
    node.left->accept(*this);
    std::cout << ", ";
    node.right->accept(*this);
    std::cout << ")\n";
}

void PrintVisitor::visit(RelationClause &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "RelationClause(" << node.name << ",\n";
    indent++;
    node.variables->accept(*this);
    indent--;
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << ")\n";
}


void PrintVisitor::visit(DatalogProgram &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "DatalogProgram(\n";
    indent++;
    node.relations->accept(*this);
    node.clauses->accept(*this);
    indent--;
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << ")\n";
}



} // namespace datalog
