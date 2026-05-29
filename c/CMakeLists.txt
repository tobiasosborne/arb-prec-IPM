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
