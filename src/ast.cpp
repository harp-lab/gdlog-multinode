
#include "../include/ast.h"
#include "../include/dynamic_dispatch.h"

#include <iostream>
#include <string>
#include <vector>

#define DEFAULT_AST_NODE_ACCEPT(CLASS_NAME)                                    \
    void CLASS_NAME::accept(DatalogASTVisitor &visitor) {                      \
        visitor.visit(*this);                                                  \
    }

namespace datalog {
// write implementation based on above commented definitions
// tokenizer has been implemented in tokenizer.cpp

// ASTNodeList
void DatalogASTNodeList::add(DatalogASTNode *node) { nodes.push_back(node); }

void DatalogASTNodeList::accept(DatalogASTVisitor &visitor) {
    visitor.visit(*this);
}

// ColumnDefinition
ColumnDefinition::ColumnDefinition(const std::string &name, ColumnType col_type)
    : name(name), col_type(col_type) {
    type = DatalogASTNodeType::COLUMN_DEFINITION;
}
DEFAULT_AST_NODE_ACCEPT(ColumnDefinition)

// RelationDefinition
RelationDefinition::RelationDefinition(const std::string &name,
                                       DatalogASTNodeList *columns)
    : name(name), columns(columns) {
    type = DatalogASTNodeType::RELATION_DEFINITION;
}
DEFAULT_AST_NODE_ACCEPT(RelationDefinition)

// HornClause
HornClause::HornClause(DatalogASTNode *head, DatalogASTNodeList *body)
    : head(head), body(body) {
    type = DatalogASTNodeType::HORN_CLAUSE;
}
DEFAULT_AST_NODE_ACCEPT(HornClause)

// MetaVariable
MetaVariable::MetaVariable(const std::string &name) : name(name) {}
DEFAULT_AST_NODE_ACCEPT(MetaVariable)

// ArithmeticExpression
ArithmeticExpression::ArithmeticExpression(ArithmeticOperator op,
                                           DatalogASTNode *left,
                                           DatalogASTNode *right)
    : op(op), left(left), right(right) {
    type = DatalogASTNodeType::ARITHMETIC_EXPRESSION;
}
DEFAULT_AST_NODE_ACCEPT(ArithmeticExpression)

// ArithmeticClause
ArithmeticClause::ArithmeticClause(MetaVariable *var, ArithmeticExpression *expr)
    : var(var), expr(expr) {
    type = DatalogASTNodeType::ARITHMETIC_CLAUSE;
}
DEFAULT_AST_NODE_ACCEPT(ArithmeticClause)

// Constant
Constant::Constant(int value) : value(value), const_type(ConstantType::INT) {
    type = DatalogASTNodeType::CONSTANT;
}
Constant::Constant(const std::string &value)
    : value(value), const_type(ConstantType::STRING) {
    type = DatalogASTNodeType::CONSTANT;
}
DEFAULT_AST_NODE_ACCEPT(Constant)

// Constraint
Constraint::Constraint(ComparisonOperator op, DatalogASTNode *left,
                       DatalogASTNode *right)
    : op(op), left(left), right(right) {
    type = DatalogASTNodeType::CONSTRAINT;
}
DEFAULT_AST_NODE_ACCEPT(Constraint)

// RelationClause
RelationClause::RelationClause(const std::string &name,
                               DatalogASTNodeList *variables)
    : name(name), variables(variables) {
    type = DatalogASTNodeType::RELATION_CLAUSE;
}
DEFAULT_AST_NODE_ACCEPT(RelationClause)

void DeclarationList::add(DatalogASTNode *node) { nodes.push_back(node); }

void HornClausesList::add(DatalogASTNode *node) { nodes.push_back(node); }

// Stratum
Stratum::Stratum(const std::string &name, DatalogASTNodeList *horn_clauses)
    : name(name), horn_clauses(horn_clauses) {
    type = DatalogASTNodeType::STRATUM;
}
DEFAULT_AST_NODE_ACCEPT(Stratum)

// DatalogProgram
DatalogProgram::DatalogProgram(DatalogASTNodeList *relations,
                               DatalogASTNodeList *stratums)
    : relations(relations), stratums(stratums) {
    type = DatalogASTNodeType::DATALOG_PROGRAM;
}
DEFAULT_AST_NODE_ACCEPT(DatalogProgram)

// Parser
Parser::Parser(const std::string &input) : tokenizer(input) {}

DatalogASTNode *Parser::parse() { return parse_program(); }

DatalogASTNode *Parser::parse_relation_definition() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    tokenizer.expect(Token{TokenType::RELATION, "relation"});
    auto name_token = tokenizer.next_token();
    auto name = name_token.str;
    auto columns = parse_column_definition_list();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new RelationDefinition(name, columns);
}

DatalogASTNode *Parser::parse_column_definition() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto name_token = tokenizer.next_token();
    auto name = name_token.str;
    auto type_token = tokenizer.next_token();
    auto type = type_token.type == TokenType::INT ? ColumnType::INT
                                                  : ColumnType::STRING;
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new ColumnDefinition(name, type);
}

DatalogASTNode *Parser::parse_horn_clause() {
    tokenizer.expect(Token{TokenType::LPRACKET, "["});
    auto head = parse_relation_clause();
    tokenizer.expect(Token{TokenType::LARROW, "<--"});
    auto body = parse_body_clause_list();
    tokenizer.expect(Token{TokenType::RPRACKET, "]"});
    return new HornClause(head, body);
}

DatalogASTNode *Parser::parse_body_clause() {
    auto peak_token = tokenizer.peak(1);
    if (peak_token.type == TokenType::LET) {
        // std::cout << "let\n";
        return parse_arithmetic_clause();
    } else if (is_comparison_token(peak_token)) {
        // std::cout << "constraint\n";
        return parse_constraint();
    } else {
        // std::cout << "relation\n";
        return parse_relation_clause();
    }
}

DatalogASTNodeList *Parser::parse_body_clause_list() {
    auto list = new DatalogASTNodeList();
    while (tokenizer.peak().type != TokenType::RPRACKET) {
        list->add(parse_body_clause());
    }
    return list;
}

DatalogASTNode *Parser::parse_meta_variable() {
    auto name_token = tokenizer.next_token();
    return new MetaVariable(name_token.str);
}

DatalogASTNode *Parser::parse_constant() {
    auto token = tokenizer.next_token();
    if (token.type == TokenType::INT) {
        return new Constant(std::stoi(token.str));
    } else {
        return new Constant(token.str);
    }
}

DatalogASTNode *Parser::parse_arithmetic_arg() {
    auto peak_token = tokenizer.peak();
    if (peak_token.type == TokenType::LPAREN) {
        return parse_arithmetic_expression();
    } else if (peak_token.type == TokenType::INT ||
               peak_token.type == TokenType::STRING) {
        return parse_constant();
    } else {
        return parse_meta_variable();
    }
}

DatalogASTNode *Parser::parse_arithmetic_expression() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto op_token = tokenizer.next_token();
    auto op = op_token.str;
    auto left = parse_meta_variable();
    auto right = parse_arithmetic_arg();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new ArithmeticExpression(op == "+"   ? ArithmeticOperator::ADD
                                    : op == "-" ? ArithmeticOperator::SUB
                                    : op == "*" ? ArithmeticOperator::MUL
                                    : op == "/" ? ArithmeticOperator::DIV
                                                : ArithmeticOperator::MOD,
                                    left, right);
}

// arithmetic_expression | constant | metavar
DatalogASTNode *Parser::parse_constraint_right() {
    auto peak_token = tokenizer.peak();
    if (peak_token.type == TokenType::LPAREN) {
        return parse_arithmetic_expression();
    } else if (peak_token.type == TokenType::INT ||
               peak_token.type == TokenType::STRING) {
        return parse_constant();
    } else {
        return parse_meta_variable();
    }
}

// Constraint ::= (comparison_operator metavar (arithmetic_expression | constant | metavar))
DatalogASTNode *Parser::parse_constraint() {
    auto cur_token = tokenizer.peak();
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto op_token = tokenizer.next_token();
    auto op = op_token.str;
    auto left = parse_meta_variable();
    auto right = parse_constraint_right();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new Constraint(op == "=="   ? ComparisonOperator::EQ
                          : op == "!=" ? ComparisonOperator::NE
                          : op == "<"  ? ComparisonOperator::LT
                          : op == "<=" ? ComparisonOperator::LE
                          : op == ">"  ? ComparisonOperator::GT
                                       : ComparisonOperator::GE,
                          left, right);
}


// arithmetic_clause ::= (let var arithmetic_expression)
DatalogASTNode *Parser::parse_arithmetic_clause() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    tokenizer.expect(Token{TokenType::LET, "let"});
    auto var = parse_meta_variable();
    // std::cout << "var: " << ((MetaVariable*)var)->name << "\n";
    auto expr = parse_arithmetic_expression();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new ArithmeticClause(static_cast<MetaVariable *>(var),
                                static_cast<ArithmeticExpression *>(expr));
}

DatalogASTNode *Parser::parse_relation_clause() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto token = tokenizer.next_token();
    auto name = token.str;
    auto variables = parse_variable_list();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new RelationClause(name, variables);
}

DatalogASTNodeList *Parser::parse_relation_definition_list() {
    auto list = new DatalogASTNodeList();
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_relation_definition());
    }
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return list;
}

DatalogASTNodeList *Parser::parse_column_definition_list() {
    auto list = new DatalogASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_column_definition());
    }
    return list;
}

DatalogASTNodeList *Parser::parse_horn_clause_list() {
    auto list = new DatalogASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_horn_clause());
    }
    return list;
}

DatalogASTNodeList *Parser::parse_variable_list() {
    auto list = new DatalogASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_meta_variable());
    }
    return list;
}

DatalogASTNodeList *Parser::parse_constant_list() {
    auto list = new DatalogASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_constant());
    }
    return list;
}

DatalogASTNodeList *Parser::parse_arithmetic_expression_list() {
    auto list = new DatalogASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_arithmetic_expression());
    }
    return list;
}

DatalogASTNodeList *Parser::parse_constraint_list() {
    auto list = new DatalogASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_constraint());
    }
    return list;
}

DatalogASTNodeList *Parser::parse_declaration_list() {
    auto list = new DatalogASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_relation_definition());
    }
    return list;
}

DatalogASTNode *Parser::parse_stratum() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    tokenizer.expect(Token{TokenType::STRATUM, "stratum"});
    auto name_token = tokenizer.next_token();
    auto name = name_token.str;
    auto horn_clauses = parse_horn_clause_list();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});    
    return new Stratum(name, horn_clauses);
}

DatalogASTNodeList *Parser::parse_stratum_list() {
    auto list = new DatalogASTNodeList();
    while (tokenizer.peak().type != TokenType::RPAREN) {
        list->add(parse_stratum());
    }
    return list;
}

// DatalogProgram ::= (declaration_list stratum_list)
DatalogASTNode *Parser::parse_program() {
    tokenizer.expect(Token{TokenType::LPAREN, "("});
    auto relations = parse_relation_definition_list();
    auto clauses = parse_stratum_list();
    tokenizer.expect(Token{TokenType::RPAREN, ")"});
    return new DatalogProgram(relations, clauses);
}

// PrintVisitor
void PrintVisitor::visit(DatalogASTNodeList &node) {
    for (auto n : node.nodes) {
        n->accept(*this);
    }
}

void PrintVisitor::visit(ColumnDefinition &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "ColumnDefinition(" << node.name << ", "
              << (node.col_type == ColumnType::INT ? "int" : "string") << ")\n";
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
    if (node.const_type == ConstantType::INT) {
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

void PrintVisitor::visit(ArithmeticClause &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "ArithmeticClause(" << node.var->name << ",\n";
    indent++;
    node.expr->accept(*this);
    indent--;
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
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

void PrintVisitor::visit(Stratum &node) {
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << "Stratum(" << node.name << ",\n";
    indent++;
    node.horn_clauses->accept(*this);
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
    node.stratums->accept(*this);
    indent--;
    for (int i = 0; i < indent; i++) {
        std::cout << "  ";
    }
    std::cout << ")\n";
}

} // namespace datalog
