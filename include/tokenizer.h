
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
    STRATUM,
    IDENTIFIER,
    INT,
    LET,
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

/**
 * @brief Checks if a token is a comparison token.
 * @param token The token to check.
 * @return True if the token is a comparison token, false otherwise.
 */
bool is_comparison_token(Token &token);

/**
 * @brief Checks if a character is a whitespace character.
 * @param c The character to check.
 * @return True if the character is a whitespace character, false otherwise.
 */
bool is_whitespace(char c);

/**
 * @brief The Tokenizer class for tokenizing input strings.
 */
class Tokenizer {
    public:
        /**
         * @brief Constructs a Tokenizer object with the given input string.
         * @param input The input string to tokenize.
         */
        Tokenizer(const std::string &input);

        /**
         * @brief Retrieves the next token from the input string.
         * @return The next token.
         */
        Token next_token();

        /**
         * @brief Retrieves the nth token from the input string without consuming it.
         * @param n The index of the token to retrieve.
         * @return The nth token.
         */
        Token peak(int n = 0);

        /**
         * @brief Expects the next token to be equal to the specified expected token.
         * @param expected The expected token.
         */
        void expect(Token expected);

        /**
         * @brief Expects the end of the input string.
         */
        void expect_eof();

        /**
         * @brief Retrieves the current position in the input string.
         * @return The current position.
         */
        size_t get_pos() const { return pos; };

    private:
        const std::string &input;
        size_t pos;
        size_t token_start;
        size_t token_end;
};

} // namespace datalog
