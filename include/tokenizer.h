
// a tiny tokenizer for datalog

// example:
// (relation foo (column1 type) (column type)))
// -->
// [LPAREN RELATION WHITESPACE IDENTIFIER WHITESPACE LPAREN IDENTIFIER
// WHITESPACE IDENTIFIER WHITESPACE RPAREN WHITESPACE LPAREN IDENTIFIER
// WHITESPACE IDENTIFIER WHITESPACE RPAREN RPAREN]
// [(foo X Y) <-- (bar X Z) (baz Z Y)]
// -->
// [LPAREN IDENTIFIER WHITESPACE IDENTIFIER WHITESPACE IDENTIFIER RPAREN
// WHITESPACE LARROW WHITESPACE LPAREN IDENTIFIER WHITESPACE IDENTIFIER
// WHITESPACE IDENTIFIER RPAREN WHITESPACE LPAREN IDENTIFIER WHITESPACE
// IDENTIFIER WHITESPACE IDENTIFIER RPAREN]

#pragma once

#include <string>

namespace datalog {

enum class TokenType {
    END,
    WHITESPACE,
    RELATION,
    LPAREN,
    RPAREN,
    LPRACKET,
    RPRACKET,
    LARROW,
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
    MOD,
};

struct Token {
    TokenType type;
    std::string str;
    int value;
};

// define whitespace
bool is_whitespace(char c);

// tokenize class
class Tokenizer {
    public:
        Tokenizer(const std::string &input);
        Token next_token();
        std::string token_string() const;
        int token_int() const;
        void expect(Token expected);
        void expect(const std::string &expected);
        void expect(int expected);
        void expect_eof();

private:
        const std::string &input;
        size_t pos;
        size_t token_start;
        size_t token_end;

};


} // namespace datalog
