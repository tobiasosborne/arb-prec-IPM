/*
 * test/test_version.c -- smoke test: arbsdp_version() is non-NULL and matches
 * the compile-time ARBSDP_VERSION macro.
 *
 * This is the first ctest target; it proves the build system wires the library
 * and header correctly before any solver code exists.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arbsdp/api.h"

int
main(void)
{
    const char *v = arbsdp_version();

    if (v == NULL) {
        fprintf(stderr, "FAIL: arbsdp_version() returned NULL\n");
        return EXIT_FAILURE;
    }

    if (strcmp(v, ARBSDP_VERSION) != 0) {
        fprintf(stderr,
                "FAIL: arbsdp_version() = \"%s\", expected \"%s\"\n",
                v, ARBSDP_VERSION);
        return EXIT_FAILURE;
    }

    printf("PASS: arbsdp_version() = \"%s\"\n", v);
    return EXIT_SUCCESS;
}
