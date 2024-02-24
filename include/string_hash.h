#pragma once

#include <cstdint>
#include <string>

inline uint32_t string_hash32(const std::string& str) {
    const uint32_t base = 2166136261u;
    const uint32_t prime = 16777619u;

    uint32_t hash = base;
    for (char c: str)
    {
        if ((int)c == 0) continue;
        hash ^= (int)c;
        hash *= prime;
    }
    return hash;
}

inline uint64_t string_hash(const std::string& str) {
    const uint64_t base = 14695981039346656037ULL;
    const uint64_t prime = 1099511628211ULL;
    const uint64_t c46 = 35184372088832ULL;

    uint64_t hash = base;
    for (char c: str)
    {
        if ((uint64_t)c == 0) continue;
        hash ^= (uint64_t)c;
        hash *= prime;
    }
    return hash % c46;
}
