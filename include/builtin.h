
#pragma once

#include "./tuple.cuh"

enum class BinaryFilterComparison {
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
};

struct TupleFilter {
    __host__ __device__ virtual bool operator()(const tuple_type &tuple) {
        bool result = true;
        for (int i = 0; i < arity; i++) {
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
                    break;
                case BinaryFilterComparison::NE:
                    result = result && (left_v != right_v);
                    break;
                case BinaryFilterComparison::LT:
                    result = result && (left_v < right_v);
                    break;
                case BinaryFilterComparison::LE:
                    result = result && (left_v <= right_v);
                    break;
                case BinaryFilterComparison::GT:
                    result = result && (left_v > right_v);
                    break;
                case BinaryFilterComparison::GE:
                    result = result && (left_v >= right_v);
                    break;
                }
            }
        }
        return result;
    };

    int arity;
    int pos;
    BinaryFilterComparison op[10];
    int left[10];
    int right[10];

    // init these field in constructor
    TupleFilter(int arity, int pos, BinaryFilterComparison op[], int left[],
                int right[])
        : arity(arity), pos(pos) {
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
    int project[10];

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
};

struct TupleGenerator {
    __host__ __device__ tuple_type operator()(const tuple_type &left,
                                              const tuple_type &right) {
        tuple_type result;
        for (int i = 0; i < arity; i++) {
            u64 left_v;
            u64 right_v;
            // all value < -16 are considered as constant
            if (left[i] >= -16) {
                left_v = left[i];
            } else {
                left_v = -left[i] - 16;
            }
            if (right[i] >= -16) {
                right_v = right[i];
            } else {
                right_v = -right[i] - 16;
            }

            switch (op[i]) {
            case BinaryArithmeticOperator::ADD:
                result[i] = left_v + right_v;
                break;
            case BinaryArithmeticOperator::SUB:
                result[i] = left_v - right_v;
                break;
            case BinaryArithmeticOperator::MUL:
                result[i] = left_v * right_v;
                break;
            case BinaryArithmeticOperator::DIV:
                result[i] = left_v / right_v;
                break;
            case BinaryArithmeticOperator::MOD:
                result[i] = left_v % right_v;
                break;
            }
        }
        return result;
    };

    int arity;
    BinaryArithmeticOperator op[10];

    TupleGenerator(int arity, BinaryArithmeticOperator op[]) : arity(arity) {
        for (int i = 0; i < arity; i++) {
            this->op[i] = op[i];
        }
    }
};
