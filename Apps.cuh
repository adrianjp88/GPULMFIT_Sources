#ifndef _APPS_H_
#define _APPS_H_

#define NOMINMAX // Must define before "#include <windows.h>" to use max() and min() in stdlib.h
#include <windows.h> 

#include <stdio.h> // for using sprintf_s

/* Variables used in this file only */
#define TempStringBufferSize 5120

#if defined(_WIN32) && defined(_MSC_VER)
typedef unsigned __int64 memsize_t;
#else
#include <stdint.h>
typedef uint64_t memsize_t;
#endif

#endif