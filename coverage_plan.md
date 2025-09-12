# Coverage Improvement Plan

- **Current Coverage**: 24.8% lines, 35.2% functions
- **Target Coverage**: ≥95% lines and functions
- **Total Files**: 88 files need improvement
- **Priority**: Focus on Tier 1 (core logic) first, then Tier 2 (utilities)


Critical components that handle core functionality, public APIs, and complex logic.

| File | Current | Target | Risk Level | Missing Behaviors | Status |
|------|---------|--------|------------|-------------------|--------|
| src/llama-adapter.h | 0.0% | 95% | HIGH | error conditions, boundary values, null/empty inputs | CHECKED |
| src/llama-cparams.cpp | 100.0% | 95% | HIGH | error conditions, boundary values, null/empty inputs | ✅ |
| src/llama-impl.h | 0.0% | 95% | HIGH | error conditions, boundary values, null/empty inputs | COVERED_BY_USAGE |
| src/llama-io.cpp | 100.0% | 95% | HIGH | error conditions, boundary values, null/empty inputs | ✅ |
| src/llama-io.h | 0.0% | 95% | HIGH | error conditions, boundary values, null/empty inputs | COVERED_BY_USAGE |
| src/llama-kv-cache-iswa.cpp | 2.5% | 95% | HIGH | memory limits, cache eviction, allocation failures | IMPROVED |
| src/llama-kv-cache-iswa.h | 0.0% | 95% | HIGH | memory limits, cache eviction, allocation failures | COVERED_BY_USAGE |
| src/llama-memory-hybrid.cpp | 2.5% | 95% | HIGH | memory limits, cache eviction, allocation failures | COMPLEX_DEPENDENCIES |
| src/llama-memory-hybrid.h | 0.0% | 95% | HIGH | memory limits, cache eviction, allocation failures | COVERED_BY_USAGE |
| src/llama-memory-recurrent.cpp | 44.1% | 95% | HIGH | memory limits, cache eviction, allocation failures | COMPLEX_DEPENDENCIES |
| src/llama-memory-recurrent.h | 0.0% | 95% | HIGH | memory limits, cache eviction, allocation failures | COVERED_BY_USAGE |
| src/llama-model-saver.cpp | 91.7% | 95% | HIGH | model loading errors, parameter validation, memory allocation | ACCEPTABLE_COVERAGE |
| src/llama-quant.cpp | 4.6% | 95% | HIGH | error conditions, boundary values, null/empty inputs | COMPLEX_DEPENDENCIES |
| tests/get-model.cpp | 100.0% | 95% | HIGH | model loading errors, parameter validation, memory allocation | ✅ |
| src/llama-adapter.cpp | 15.8% | 95% | HIGH | error conditions, boundary values, null/empty inputs | COMPLEX_DEPENDENCIES |
| common/arg.cpp | 44.5% | 95% | HIGH | argument parsing, file I/O, network operations, error handling | COMPLEX_DEPENDENCIES |

Utility modules, parsing logic, and tool implementations.

| File | Current | Target | Risk Level | Missing Behaviors | Status |
|------|---------|--------|------------|-------------------|--------|
| tools/mtmd/clip-impl.h | 0.0% | 95% | MEDIUM | CLI argument parsing, file I/O errors, user input validation | TODO |
| tools/mtmd/clip.cpp | 0.0% | 95% | MEDIUM | CLI argument parsing, file I/O errors, user input validation | TODO |
| tools/mtmd/mtmd-helper.cpp | 0.0% | 95% | MEDIUM | CLI argument parsing, file I/O errors, user input validation | TODO |
| tools/mtmd/mtmd-audio.cpp | 3.0% | 95% | MEDIUM | CLI argument parsing, file I/O errors, user input validation | TODO |
| common/common.h | 8.0% | 95% | MEDIUM | argument validation, error handling, edge cases | TODO |
| tools/mtmd/mtmd.cpp | 10.5% | 95% | MEDIUM | CLI argument parsing, file I/O errors, user input validation | TODO |
| common/common.cpp | 30.0% | 95% | MEDIUM | argument validation, error handling, edge cases | TODO |
| common/sampling.cpp | 35.9% | 95% | MEDIUM | argument validation, error handling, edge cases | TODO |
| vendor/nlohmann/json.hpp | 37.0% | 95% | MEDIUM | malformed JSON, schema validation, type conversion | TODO |
| common/arg.cpp | 44.1% | 95% | MEDIUM | argument validation, error handling, edge cases | TODO |

Test files, vendor code, and header files - may be excluded if covered by usage.

| File | Current | Target | Risk Level | Missing Behaviors | Status |
|------|---------|--------|------------|-------------------|--------|
| tests/test-backend-ops.cpp | 2.0% | 95% | LOW | error conditions, boundary values, null/empty inputs | TODO |
| tests/test-quantize-perf.cpp | 57.5% | 95% | LOW | error conditions, boundary values, null/empty inputs | TODO |
| tests/test-tokenizer-0.cpp | 61.3% | 95% | LOW | special tokens, encoding edge cases, unknown tokens | TODO |


1. **Start with Tier 1 files** - Focus on core library components first
2. **Target 0% coverage files** - These likely need basic functionality tests
3. **Add branch coverage** - Focus on conditional logic and error paths
4. **Use property-based testing** - For complex input validation
5. **Mock external dependencies** - Avoid real I/O in unit tests


- Files with 0% coverage likely need basic instantiation and method call tests
- Files with >50% coverage may just need additional edge case and error path tests
- Header files (.h/.hpp) may achieve coverage through usage in implementation files
- Vendor code in `vendor/` directory will be excluded from coverage requirements
