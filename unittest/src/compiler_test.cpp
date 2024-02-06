
// datalog compiler frontend test

#include "../../include/tokenizer.h"

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
        std::cout << static_cast<int>(token.type) << " " << token.str << std::endl;
        // assert token type
        assert(token.type == expected_types[0]);
    }   
}

// test tokenizer for inference rule
void test_tokenizer_infer() {
    std::string input = "(foo X Y) <-- (bar X Z) (baz Z Y)";
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
        std::cout << static_cast<int>(token.type) << " " << token.str << std::endl;
        // assert token type
        assert(token.type == expected_types[0]);
    }   
}

void test_tokenizer() {
    test_tokenizer_decl();
    test_tokenizer_infer();
}

int main() {
    std::cout << "Begin Compiler Test" << std::endl;
    test_tokenizer();
    return 0;
}
