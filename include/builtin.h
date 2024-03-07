
#pragma once

#include "./tuple.cuh"

#define H_STR(x) (-(long)x - 15)
#define EMPTY_COLUMN 0

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
            long left = this->left[i];
            long right = this->right[i];
            u64 left_v;
            u64 right_v;
            // all value < -16 are considered as constant
            if (left >= 0 && left < 10) {
                left_v = tuple[left];
            } else if (left >= MAX_ARITY) {
                if (left == C_ZERO) {
                    left_v = 0;
                } else {
                    left_v = left;
                }
            } else {
                left_v = -left - 16;
            }
            if (right >= 0 && right < MAX_ARITY) {
                right_v = tuple[right];
            } else if (right >= MAX_ARITY) {
                if (right == C_ZERO) {
                    right_v = 0;
                } else {
                    right_v = right;
                }
            } else {
                right_v = -right - 16;
            }
            // printf("left: %ld, leftv: %ld, right: %ld , rightv : %ld \n",
            // left, left_v, right, right_v);
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

    int arity = 0;
    int pos;
    BinaryFilterComparison op[MAX_ARITY];
    long left[MAX_ARITY];
    long right[MAX_ARITY];

    TupleFilter() = default;

    // init these field in constructor
    TupleFilter(int arity, std::vector<BinaryFilterComparison> op,
                std::vector<long> left, std::vector<long> right)
        : arity(arity) {
        for (int i = 0; i < arity; i++) {
            this->op[i] = op[i];
            this->left[i] = left[i];
            this->right[i] = right[i];
        }
    }

    TupleFilter(std::vector<BinaryFilterComparison> op, std::vector<long> left,
                std::vector<long> right) {
        arity = op.size();
        for (int i = 0; i < arity; i++) {
            this->op[i] = op[i];
            this->left[i] = left[i];
            this->right[i] = right[i];
        }
    }
};

struct TupleJoinFilter {
    __host__ __device__ bool operator()(const tuple_type tp_inner,
                                        tuple_type tp_outer) {
        bool result = true;
        for (int i = 0; i < arity; i++) {
            int left = this->left[i];
            int right = this->right[i];
            u64 left_v = 0;
            u64 right_v = 0;
            // all value < -16 are considered as constant
            if (left >= inner_arity && left < MAX_ARITY) {
                left_v = tp_outer[left - inner_arity];
            } else if (left >= 0 && left < inner_arity) {
                left_v = tp_inner[left];
            } else if (left >= MAX_ARITY) {
                if (left == C_ZERO) {
                    left_v = 0;
                } else {
                    left_v = left;
                }
            } else {
                left_v = -left - 16;
            }
            if (right >= inner_arity) {
                right_v = tp_outer[right - inner_arity];
            } else if (right >= 0 && right < inner_arity) {
                right_v = tp_inner[right];
            } else if (right >= MAX_ARITY) {
                if (right == C_ZERO) {
                    right_v = 0;
                } else {
                    right_v = right;
                }

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

    int arity = 0;
    int inner_arity;
    int pos;
    BinaryFilterComparison op[MAX_ARITY];
    int left[MAX_ARITY];
    int right[MAX_ARITY];

    TupleJoinFilter() = default;

    // init these field in constructor
    TupleJoinFilter(int arity, int inner_arity,
                    std::vector<BinaryFilterComparison> op,
                    std::vector<int> left, std::vector<int> right)
        : arity(arity), inner_arity(inner_arity) {
        for (int i = 0; i < arity; i++) {
            this->op[i] = op[i];
            this->left[i] = left[i];
            this->right[i] = right[i];
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
        column_type result[MAX_ARITY];
        for (int i = 0; i < arity; i++) {
            auto cur_op = op[i];
            if (cur_op == BinaryArithmeticOperator::EMPTY) {
                result[i] = tuple[i];
                continue;
            }
            int left = this->left[i];
            int right = this->right[i];
            u64 left_v;
            u64 right_v;
            // all value < -16 are considered as constant
            if (left >= 0 && left < MAX_ARITY) {
                left_v = tuple[left];
            } else if (left >= MAX_ARITY) {
                if (left == C_ZERO) {
                    left_v = 0;
                } else {
                    left_v = left;
                }
            } else {
                left_v = -left - 16;
            }
            if (right >= 0 && right < MAX_ARITY) {
                right_v = tuple[right];
            } else if (right >= MAX_ARITY) {
                if (right == C_ZERO) {
                    right_v = 0;
                } else {
                    right_v = right;
                }
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
    long left[MAX_ARITY];
    long right[MAX_ARITY];

    TupleArithmetic(int arity, std::vector<BinaryArithmeticOperator> op,
                    std::vector<long> left, std::vector<long> right)
        : arity(arity) {
        for (int i = 0; i < arity; i++) {
            this->op[i] = op[i];
            this->left[i] = left[i];
            this->right[i] = right[i];
        }
    }

    TupleArithmetic(std::vector<BinaryArithmeticOperator> &op,
                    std::vector<long> &left, std::vector<long> &right) {
        arity = op.size();
        for (int i = 0; i < arity; i++) {
            this->op[i] = op[i];
            this->left[i] = left[i];
            this->right[i] = right[i];
        }
    }
};

struct TupleArithmeticSingle {
    __host__ __device__ tuple_type operator()(const tuple_type tuple) {
        column_type result;
        if (op == BinaryArithmeticOperator::EMPTY) {
            return tuple;
        }

        column_type left_v = tuple[left];
        column_type right_v = tuple[right];
        // NOTE: left can only be columns here!
        if (right >= 0 && right < MAX_ARITY) {
            right_v = tuple[right];
        } else if (right >= MAX_ARITY) {
            if (right == C_ZERO) {
                right_v = 0;
            } else {
                right_v = right;
            }
        } else {
            right_v = -right - 16;
        }
        switch (op) {
        case BinaryArithmeticOperator::ADD:
            result = left_v + right_v;
            break;
        case BinaryArithmeticOperator::SUB:
            result = left_v - right_v;
            break;
        case BinaryArithmeticOperator::MUL:
            result = left_v * right_v;
            break;
        case BinaryArithmeticOperator::DIV:
            result = left_v / right_v;
            break;
        case BinaryArithmeticOperator::MOD:
            result = left_v % right_v;
            break;
        case BinaryArithmeticOperator::EMPTY:
            result = tuple[left];
            break;
        }
        tuple[left] = result;
        return tuple;
    };

    int arity;
    BinaryArithmeticOperator op;
    long left;
    long right;

    // TupleArithmeticSingle(std::vector<BinaryArithmeticOperator> &op,
    //                       std::vector<long> &left, std::vector<long> &right) {
    //     arity = op.size();
    //     this->op = op[0];
    //     this->left = left[0];
    //     this->right = right[0];
    // }

    TupleArithmeticSingle(std::vector<BinaryArithmeticOperator> op,
                          std::vector<long> left, std::vector<long> right) {
        arity = op.size();
        this->op = op[0];
        this->left = left[0];
        this->right = right[0];
    }
};
