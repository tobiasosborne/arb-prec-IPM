cmake_minimum_required(VERSION 3.16)
project(arbsdp LANGUAGES C)

# ---- C standard ----------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ---- Options -------------------------------------------------------------
option(ENABLE_ASAN
    "Build with AddressSanitizer + UBSan (-fsanitize=address,undefined)"
    OFF)

# ---- Compiler flags ------------------------------------------------------
# Required flags per CLAUDE.md / FLINT_NOTES.md.
# -Wstrict-prototypes is C-only; guard for safety.
add_compile_options(
    -Wall -Wextra -Wpedantic -Wshadow
    $<$<COMPILE_LANGUAGE:C>:-Wstrict-prototypes>
    -O2 -g
)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

# ---- Dependencies --------------------------------------------------------
# FLINT 3.x: Arb is folded in.  Fail hard if not found (PRD §5.4 — the lib
# must build stand-alone against FLINT/GMP/MPFR and nothing else).
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLINT REQUIRED flint)

# GMP and MPFR have no .pc files on this machine; link by name as documented
# in FLINT_NOTES.md (link line: -lflint -lmpfr -lgmp).
# We leave them as plain library names; CMake finds them via the linker's
# default search path.

# ---- Library target ------------------------------------------------------
file(GLOB ARBSDP_SOURCES "src/*.c")

add_library(arbsdp SHARED ${ARBSDP_SOURCES})

target_include_directories(arbsdp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${FLINT_INCLUDE_DIRS}
)

target_link_libraries(arbsdp
    PUBLIC
        # The installed flint.pc (FLINT 3.0.1) omits -lflint from its own Libs:
        # line, listing only -lgmp -lmpfr; ${FLINT_LIBRARIES} therefore does NOT
        # contain flint.  Link it explicitly to match the documented link line
        # -lflint -lmpfr -lgmp (FLINT_NOTES.md).
        flint
        ${FLINT_LIBRARIES}
        mpfr
        gmp
)

target_link_directories(arbsdp
    PUBLIC
        ${FLINT_LIBRARY_DIRS}
)

# Pass any compile flags pkg-config reports (e.g. -pthread on some distros).
target_compile_options(arbsdp PRIVATE ${FLINT_CFLAGS_OTHER})

# ---- Tests ---------------------------------------------------------------
enable_testing()

add_executable(test_version test/test_version.c)

target_include_directories(test_version
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(test_version PRIVATE arbsdp)

add_test(NAME test_version COMMAND test_version)

add_executable(test_svec test/test_svec.c)

target_include_directories(test_svec
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

target_link_libraries(test_svec PRIVATE arbsdp)

add_test(NAME test_svec COMMAND test_svec)

add_executable(test_linalg test/test_linalg.c)

target_include_directories(test_linalg
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

target_link_libraries(test_linalg PRIVATE arbsdp)

add_test(NAME test_linalg COMMAND test_linalg)

add_executable(test_certify test/test_certify.c)

target_include_directories(test_certify
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

target_link_libraries(test_certify PRIVATE arbsdp)

add_test(NAME test_certify COMMAND test_certify)

add_executable(test_ntscaling test/test_ntscaling.c)

target_include_directories(test_ntscaling
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

target_link_libraries(test_ntscaling PRIVATE arbsdp)

add_test(NAME test_ntscaling COMMAND test_ntscaling)

add_executable(test_io test/test_io.c)

target_include_directories(test_io
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

# The golden fixtures live at <repo>/golden; pass an absolute path so the test
# is independent of the run directory (ctest runs in the build tree).
target_compile_definitions(test_io
    PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)

target_link_libraries(test_io PRIVATE arbsdp)

add_test(NAME test_io COMMAND test_io)

add_executable(test_iterate test/test_iterate.c)

target_include_directories(test_iterate
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

# The iterate tests load golden fixtures (sdp_sqrt2); pass an absolute path so
# the test is independent of the run directory (as test_io does).
target_compile_definitions(test_iterate
    PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)

target_link_libraries(test_iterate PRIVATE arbsdp)

add_test(NAME test_iterate COMMAND test_iterate)

add_executable(test_schur test/test_schur.c)

target_include_directories(test_schur
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

target_link_libraries(test_schur PRIVATE arbsdp)

add_test(NAME test_schur COMMAND test_schur)

add_executable(test_regularize test/test_regularize.c)

target_include_directories(test_regularize
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

target_link_libraries(test_regularize PRIVATE arbsdp)

add_test(NAME test_regularize COMMAND test_regularize)

add_executable(test_direction test/test_direction.c)

target_include_directories(test_direction
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

# The integration test (mu decreases) loads the sdp_sqrt2 golden fixture; pass an
# absolute path so the test is independent of the run directory (as test_io does).
target_compile_definitions(test_direction
    PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)

target_link_libraries(test_direction PRIVATE arbsdp)

add_test(NAME test_direction COMMAND test_direction)

add_executable(test_solve test/test_solve.c)

target_include_directories(test_solve
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

# The end-to-end solve tests load the golden fixtures and compare the recovered
# objective to the analytic 65-digit optima; pass an absolute golden path so the
# test is independent of the run directory (as test_io / test_direction do).
target_compile_definitions(test_solve
    PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)

# test_solve calls log10 directly (digit-accuracy diagnostic) -> needs libm.
target_link_libraries(test_solve PRIVATE arbsdp m)

add_test(NAME test_solve COMMAND test_solve)

add_executable(test_infeasible test/test_infeasible.c)

target_include_directories(test_infeasible
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

# test_infeasible loads the feasible golden (max_eigenvalue_2x2) for the
# no-misfire regression; pass an absolute golden path so the test is
# independent of the run directory (as test_io / test_solve do).  It also
# writes embedded infeasible .dat-s specs to /tmp.
target_compile_definitions(test_infeasible
    PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)

target_link_libraries(test_infeasible PRIVATE arbsdp)

add_test(NAME test_infeasible COMMAND test_infeasible)

# The rigorous Farkas infeasibility certificate gate (bead arb-prec-IPM-b20):
# exercises arbsdp_verify_primal_infeasible / arbsdp_verify_dual_infeasible and the
# unified arbsdp_certify dispatcher against the 6 infeasible/unbounded goldens plus
# a feasible no-misfire case.  Hardcodes golden names (no golden_provenance.h);
# passes an absolute GOLDEN_DIR like test_infeasible.  Links libm for tau_mid_d's
# arf_get_d diagnostic (no log10, but keep m for parity with the other golden tests).
add_executable(test_farkas test/test_farkas.c)

target_include_directories(test_farkas
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

target_compile_definitions(test_farkas
    PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)

target_link_libraries(test_farkas PRIVATE arbsdp m)

add_test(NAME test_farkas COMMAND test_farkas)

add_executable(test_precision test/test_precision.c)

target_include_directories(test_precision
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
)

# test_precision loads the golden fixtures (sdp_sqrt2, trivial_2x2) and compares
# the adaptively-recovered objective to the analytic golden; pass an absolute
# golden path so the test is independent of the run directory (as test_solve does).
target_compile_definitions(test_precision
    PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)

# test_precision calls log10/log2/pow (digit-accuracy diagnostics) -> needs libm.
target_link_libraries(test_precision PRIVATE arbsdp m)

add_test(NAME test_precision COMMAND test_precision)

# The GATED golden-master runner (bead arb-prec-IPM-epic.1): the primary Layer-0
# robustness gate.  Compiles the provenance scanner into the executable and reads
# golden_discover(GOLDEN_DIR, ...).  Mirrors test_io's absolute GOLDEN_DIR compile
# definition; adds the test/ dir to the include path for golden_provenance.h and
# links libm for log10 (digits_matched diagnostic).
add_executable(test_golden test/test_golden.c test/golden_provenance.c)

target_include_directories(test_golden
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/test
)

target_compile_definitions(test_golden
    PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)

target_link_libraries(test_golden PRIVATE arbsdp m)

add_test(NAME test_golden COMMAND test_golden)

# The rigorous bracket gate (bead arb-prec-IPM-b19): THE PRODUCT.  Solves each
# golden, certifies [lb,ub] in ball arithmetic (certify.c), and asserts the
# 65-digit analytic optimum is inside (RIGOR; a bracket that excludes opt is P0).
# Reuses golden_provenance.c (like test_golden) for the per-case optimal_value;
# absolute GOLDEN_DIR; test/ include path for golden_provenance.h; libm for log10.
add_executable(test_certify_bracket test/test_certify_bracket.c test/golden_provenance.c)

target_include_directories(test_certify_bracket
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${FLINT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/test
)

target_compile_definitions(test_certify_bracket
    PRIVATE
        GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../golden"
)

target_link_libraries(test_certify_bracket PRIVATE arbsdp m)

add_test(NAME test_certify_bracket COMMAND test_certify_bracket)
