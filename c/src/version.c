/*
 * src/version.c -- implement arbsdp_version()
 *
 * PRD §5.1, §5.3: minimal versioned ABI.
 */

#include "arbsdp/api.h"

const char *
arbsdp_version(void)
{
    return ARBSDP_VERSION;
}
