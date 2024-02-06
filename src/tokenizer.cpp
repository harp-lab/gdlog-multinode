#include "../include/tokenizer.h"
#include <stdexcept>

namespace datalog {

// define whitespace
bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// tokenize class
Tokenizer::Tokenizer(const std::string &input)
    : input(input), pos(0), token_start(0), token_end(0) {}

Token Tokenizer::next_token() {
    if (pos == input.size()) {
        return Token(TokenType::END, "");
    }
    char c = input[pos];
    while (is_whitespace(c)) {
        pos++;
        if (pos == input.size()) {
            return Token(TokenType::END, "");
        }
        c = input[pos];
    }
    if (c == '(') {
        pos++;
        return Token(TokenType::LPAREN, "(");
    } else if (c == ')') {
        pos++;
        return Token(TokenType::RPAREN, ")");
    } else if (c == '[') {
        pos++;
        return Token(TokenType::LPRACKET, "[");
    } else if (c == ']') {
        pos++;
        return Token(TokenType::RPRACKET, "]");
    } else if (c == '<') {
        pos++;
        if (pos < input.size() && input[pos] == '=') {
            pos++;
            return Token(TokenType::LE, "<=");
        } else if (pos + 1 < input.size() && input[pos] == '-' &&
                   input[pos + 1] == '-') {
            pos++;
            pos++;
            return Token(TokenType::LARROW, "<--");
        } else {
            return Token(TokenType::LT, "<");
        }
    } else if (c == '>') {
        pos++;
        if (pos < input.size() && input[pos] == '=') {
            pos++;
            return Token(TokenType::GE, ">=");
        } else {
            return Token(TokenType::GT, ">");
        }
    } else if (c == '+') {
        pos++;
        return Token(TokenType::ADD, "+");
    } else if (c == '-') {
        pos++;
        return Token(TokenType::SUB, "-");
    } else if (c == '*') {
        pos++;
        return Token(TokenType::MUL, "*");
    } else if (c == '/') {
        pos++;
        return Token(TokenType::DIV, "/");
    } else if (c == '%') {
        pos++;
        return Token(TokenType::MOD, "%");
    } else if (c == '=') {
        pos++;
        return Token(TokenType::EQ, "=");
    } else if (c == '!') {
        pos++;
        if (pos < input.size() && input[pos] == '=') {
            pos++;
            return Token(TokenType::NE, "!=");
        } else {
            throw std::runtime_error("unexpected character");
        }
    } else if (c == '"') {
        token_start = pos;
        pos++;
        while (pos < input.size() && input[pos] != '"') {
            pos++;
        }
        if (pos == input.size()) {
            throw std::runtime_error("unterminated string");
        }
        token_end = pos;
        pos++;
        return Token(TokenType::STRING,
                     input.substr(token_start, token_end - token_start));
    } else if (c == 'r' && pos + 1 < input.size() && input[pos + 1] == 'e' &&
               pos + 2 < input.size() && input[pos + 2] == 'l' &&
               pos + 3 < input.size() && input[pos + 3] == 'a' &&
               pos + 4 < input.size() && input[pos + 4] == 't' &&
               pos + 5 < input.size() && input[pos + 5] == 'i' &&
               pos + 6 < input.size() && input[pos + 6] == 'o' &&
               pos + 7 < input.size() && input[pos + 7] == 'n') {
        pos += 8;
        return Token(TokenType::RELATION, "relation");
    } else if (c >= '0' && c <= '9') {
        token_start = pos;
        while (pos < input.size() && input[pos] >= '0' && input[pos] <= '9') {
            pos++;
        }
        token_end = pos;
        return Token(TokenType::INT,
                     input.substr(token_start, token_end - token_start));
    } else if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z') {
        token_start = pos;
        while (pos < input.size() && (input[pos] >= 'a' && input[pos] <= 'z' ||
                                      input[pos] >= 'A' && input[pos] <= 'Z')) {
            pos++;
        }
        token_end = pos;
        return Token(TokenType::IDENTIFIER,
                     input.substr(token_start, token_end - token_start));
    } else {
        throw std::runtime_error("unexpected character");
    }
}

} // namespace datalog
