
// datalog compiler frontend test

#include "../../include/tokenizer.h"
#include "../../include/ast.h"

#include <iostream>
#include <vector>
#include <cassert>

// unit test for tokenizer
// cover relation definition, and inference rule
void test_tokenizer_decl() {
    std::string input = "(relation foo (column1 type) (column type))";
    // expected token type vector (not including whitespace)
    std::vector<datalog::TokenType> expected_types = {
        datalog::TokenType::LPAREN, datalog::TokenType::RELATION,
        datalog::TokenType::IDENTIFIER, datalog::TokenType::LPAREN,
        datalog::TokenType::IDENTIFIER, datalog::TokenType::IDENTIFIER,
        datalog::TokenType::RPAREN, datalog::TokenType::LPAREN,
        datalog::TokenType::IDENTIFIER, datalog::TokenType::IDENTIFIER,
        datalog::TokenType::RPAREN, datalog::TokenType::RPAREN};

    datalog::Tokenizer tokenizer(input);
    std::vector<datalog::Token> tokens;
    while (true) {
        datalog::Token token = tokenizer.next_token();
        tokens.push_back(token);
        if (token.type == datalog::TokenType::END) {
            break;
        }
    }
    for (auto token : tokens) {
        // std::cout << static_cast<int>(token.type) << " " << token.str << std::endl;
        // assert token type
        assert(token.type == expected_types[0]);
    }   
}

// test tokenizer for inference rule
void test_tokenizer_infer() {
    std::string input = "[(foo X Y) <-- (bar X Z) (baz Z Y)]";
    // expected token type vector (not including whitespace)
    std::vector<datalog::TokenType> expected_types = {
        datalog::TokenType::LPAREN, datalog::TokenType::IDENTIFIER,
        datalog::TokenType::IDENTIFIER, datalog::TokenType::IDENTIFIER,
        datalog::TokenType::RPAREN, datalog::TokenType::LARROW,
        datalog::TokenType::LPAREN, datalog::TokenType::IDENTIFIER,
        datalog::TokenType::IDENTIFIER, datalog::TokenType::IDENTIFIER,
        datalog::TokenType::RPAREN, datalog::TokenType::LPAREN,
        datalog::TokenType::IDENTIFIER, datalog::TokenType::IDENTIFIER,
        datalog::TokenType::IDENTIFIER, datalog::TokenType::RPAREN};

    datalog::Tokenizer tokenizer(input);
    std::vector<datalog::Token> tokens;
    while (true) {
        datalog::Token token = tokenizer.next_token();
        tokens.push_back(token);
        if (token.type == datalog::TokenType::END) {
            break;
        }
    }
    for (auto token : tokens) {
        // std::cout << static_cast<int>(token.type) << " " << token.str << std::endl;
        // assert token type
        assert(token.type == expected_types[0]);
    }   
}

void test_tokenizer() {
    test_tokenizer_decl();
    test_tokenizer_infer();
}


void test_parser() {
    std::string input =
        "("
        "((relation foo (column1 int) (column int)))"
        "[(foo X Y) <-- (bar X Z) (baz Z Y)]"
        ")";
    datalog::Parser parser(input);
    datalog::ASTNode *node = parser.parse();
    assert(node->type == datalog::ASTNodeType::DATALOG_PROGRAM);
    datalog::DatalogProgram *program = dynamic_cast<datalog::DatalogProgram *>(node);
    assert(program->relation_definitions->size() == 1);
    assert(program->horn_clauses->size() == 1);
    datalog::RelationDefinition *relation_definition = dynamic_cast<datalog::RelationDefinition *>(program->relations->at(0));
    assert(relation_definition->name == "foo");
    assert(relation_definition->columns->size() == 2);
    datalog::ColumnDefinition *column1 = dynamic_cast<datalog::ColumnDefinition *>(relation_definition->columns->at(0));
    assert(column1->name == "column1");
    assert(column1->type == "int");
    datalog::HornClause *horn_clause = dynamic_cast<datalog::HornClause *>(program->clauses->at(0));
    assert(horn_clause->head->name == "foo");
    assert(horn_clause->head->arguments->size() == 2);
    assert(horn_clause->body->size() == 2);
    datalog::RelationClause *relation_clause = dynamic_cast<datalog::RelationClause *>(horn_clause->body->at(0));
    assert(relation_clause->name == "bar");
    assert(relation_clause->arguments->size() == 3);
    relation_clause = dynamic_cast<datalog::RelationClause *>(horn_clause->body->at(1));
    assert(relation_clause->name == "baz");
    assert(relation_clause->arguments->size() == 3);
    std::cout << "Parser Test Passed" << std::endl;
}

int main() {
    std::cout << "Begin Compiler Test" << std::endl;
    test_tokenizer();
    test_parser();
    return 0;
}
