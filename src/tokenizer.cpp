#include "../include/tokenizer.h"
#include <format>
#include <stdexcept>

namespace datalog {

// define whitespace
bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool is_comparison_token(Token &token) {
    return token.type == TokenType::EQ || token.type == TokenType::NE ||
           token.type == TokenType::LT || token.type == TokenType::LE ||
           token.type == TokenType::GT || token.type == TokenType::GE;
}

// tokenize class
Tokenizer::Tokenizer(const std::string &input)
    : input(input), pos(0), token_start(0), token_end(0) {}

Token Tokenizer::next_token() {
    if (pos == input.size()) {
        return Token(TokenType::END, "");
    }
    char c = input[pos];
    if (c == '"') {
        token_start = pos;
        pos++;
        while (pos < input.size() && input[pos] != '"') {
            pos++;
        }
        if (pos == input.size()) {
            throw std::runtime_error(
                std::format("At {} unexpected end of string", pos));
        }
        token_end = pos;
        pos++;
        return Token(TokenType::STRING,
                     input.substr(token_start, token_end - token_start));
    }
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
        if (input[pos] == '=') {
            pos = pos + 2;
            return Token(TokenType::EQ, "=");
        }
    } else if (c == '!') {
        pos++;
        if (pos < input.size() && input[pos] == '=') {
            pos++;
            return Token(TokenType::NE, "!=");
        } else {
            // print what is unexpected character
            throw std::runtime_error(
                std::format("At {} unexpected character : {}", pos, c));
        }
    } else if (c == 'r' && pos + 1 < input.size() && input[pos + 1] == 'e' &&
               pos + 2 < input.size() && input[pos + 2] == 'l' &&
               pos + 3 < input.size() && input[pos + 3] == 'a' &&
               pos + 4 < input.size() && input[pos + 4] == 't' &&
               pos + 5 < input.size() && input[pos + 5] == 'i' &&
               pos + 6 < input.size() && input[pos + 6] == 'o' &&
               pos + 7 < input.size() && input[pos + 7] == 'n') {
        pos += 8;
        return Token(TokenType::RELATION, "relation");
    } else if (c == 's' && pos + 1 < input.size() && input[pos + 1] == 't' &&
               pos + 2 < input.size() && input[pos + 2] == 'r' &&
               pos + 3 < input.size() && input[pos + 3] == 'a' &&
               pos + 4 < input.size() && input[pos + 4] == 't' &&
               pos + 5 < input.size() && input[pos + 5] == 'u' &&
               pos + 6 < input.size() && input[pos + 6] == 'm') {
        pos += 7;
        return Token(TokenType::STRATUM, "stratum");
    } else if (c == 'l' && pos + 1 < input.size() && input[pos + 1] == 'e' &&
               pos + 2 < input.size() && input[pos + 2] == 't') {
        pos += 3;
        return Token(TokenType::LET, "let");
    } else if (c >= '0' && c <= '9') {
        token_start = pos;
        while (pos < input.size() && input[pos] >= '0' && input[pos] <= '9') {
            pos++;
        }
        token_end = pos;
        return Token(TokenType::INT,
                     input.substr(token_start, token_end - token_start));
    } else if (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '@') {
        token_start = pos;
        while (pos < input.size() && (input[pos] >= 'a' && input[pos] <= 'z' ||
                                      input[pos] >= 'A' && input[pos] <= 'Z' ||
                                      input[pos] >= '0' && input[pos] <= '9' ||
                                      input[pos] == '_' || input[pos] == '-' ||
                                      input[pos] == '@' || input[pos] == '~')) {
            pos++;
        }
        token_end = pos;
        return Token(TokenType::IDENTIFIER,
                     input.substr(token_start, token_end - token_start));
    } else {
        throw std::runtime_error(
            std::format("At {} unexpected character : {}", pos, c));
    }
}

void Tokenizer::expect(Token expected) {
    Token token = next_token();
    if (token.type != expected.type || token.str != expected.str) {
        throw std::runtime_error(
            std::format("At {} expected token {} but got {}", pos, expected.str,
                        token.str));
    }
}

void Tokenizer::expect_eof() {
    Token token = next_token();
    if (token.type != TokenType::END) {
        throw std::runtime_error(std::format(
            "At {} expected end of file but got {}", pos, token.str));
    }
}

Token Tokenizer::peak(int n) {
    size_t old_pos = pos;
    Token token;
    for (int i = 0; i <= n; i++) {
        token = next_token();
    }
    pos = old_pos;
    return token;
}

} // namespace datalog
