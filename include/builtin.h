
#pragma once

#include "./tuple.cuh"

enum class BinaryFilterComparison {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    EMPTY,
};

struct TupleFilter {
    __host__ __device__ bool operator()(const tuple_type tuple) {
        bool result = true;
        for (int i = 0; i < arity; i++) {
            int left = this->left[i];
            int right = this->right[i];
            u64 left_v = tuple[left];
            u64 right_v = tuple[right];
            // all value < -16 are considered as constant
            if (left >= -16) {
                left_v = tuple[left];
            } else {
                left_v = -left - 16;
            }
            if (right >= -16) {
                right_v = tuple[right];
            } else {
                right_v = -right - 16;
            }

            switch (op[i]) {
            case BinaryFilterComparison::EQ:
                result = result && (left_v == right_v);
                continue;
            case BinaryFilterComparison::NE:
                result = result && (left_v != right_v);
                continue;
            case BinaryFilterComparison::LT:
                result = result && (left_v < right_v);
                continue;
            case BinaryFilterComparison::LE:
                result = result && (left_v <= right_v);
                continue;
            case BinaryFilterComparison::GT:
                result = result && (left_v > right_v);
                continue;
            case BinaryFilterComparison::GE:
                result = result && (left_v >= right_v);
                continue;
            case BinaryFilterComparison::EMPTY:
                continue;
            }
        }

        return result;
    };

    int arity;
    int pos;
    BinaryFilterComparison op[MAX_ARITY];
    int left[MAX_ARITY];
    int right[MAX_ARITY];

    // init these field in constructor
    TupleFilter(int arity, std::vector<BinaryFilterComparison> &op,
                std::vector<int> &left, std::vector<int> &right)
        : arity(arity) {
        for (int i = 0; i < arity; i++) {
            this->op[i] = op[i];
            this->left[i] = left[i];
            this->right[i] = right[i];
        }
    }
};

struct TupleProjector {
    __host__ __device__ tuple_type operator()(const tuple_type &tuple) {
        tuple_type result;
        for (int i = 0; i < arity; i++) {
            result[i] = tuple[project[i]];
        }
        return result;
    };

    int arity;
    int project[MAX_ARITY];

    TupleProjector(int arity, int project[]) : arity(arity) {
        for (int i = 0; i < arity; i++) {
            this->project[i] = project[i];
        }
    }
};

enum BinaryArithmeticOperator {
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    EMPTY,
};

struct TupleArithmetic {
    __host__ __device__ tuple_type operator()(const tuple_type tuple) {
        tuple_type result;
        for (int i = 0; i < arity; i++) {
            auto cur_op = op[i];
            if (cur_op == BinaryArithmeticOperator::EMPTY) {
                result[i] = tuple[i];
                continue;
            }
            int left = this->left[i];
            int right = this->right[i];
            u64 left_v = tuple[left];
            u64 right_v = tuple[right];
            // all value < -16 are considered as constant
            if (left >= -16) {
                left_v = tuple[left];
            } else {
                left_v = -left - 16;
            }
            if (right >= -16) {
                right_v = tuple[right];
            } else {
                right_v = -right - 16;
            }
            switch (cur_op) {
            case BinaryArithmeticOperator::ADD:
                result[i] = left_v + right_v;
                continue;
            case BinaryArithmeticOperator::SUB:
                result[i] = left_v - right_v;
                continue;
            case BinaryArithmeticOperator::MUL:
                result[i] = left_v * right_v;
                continue;
            case BinaryArithmeticOperator::DIV:
                result[i] = left_v / right_v;
                continue;
            case BinaryArithmeticOperator::MOD:
                result[i] = left_v % right_v;
                continue;
            case BinaryArithmeticOperator::EMPTY:
                result[i] = tuple[i];
                continue;
            }
        }
        for (int i = 0; i < arity; i++) {
            tuple[i] = result[i];
        }
        return tuple;
    };

    int arity;
    BinaryArithmeticOperator op[MAX_ARITY];
    int left[MAX_ARITY];
    int right[MAX_ARITY];

    TupleArithmetic(int arity, std::vector<BinaryArithmeticOperator> &op,
                    std::vector<int> &left, std::vector<int> &right)
        : arity(arity) {
        for (int i = 0; i < arity; i++) {
            this->op[i] = op[i];
            this->left[i] = left[i];
            this->right[i] = right[i];
        }
    }
};
