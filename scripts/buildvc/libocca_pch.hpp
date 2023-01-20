// libocca_pch.h
// add headers to be precompiled
// 2021/10/05
// /w: 4068;4244;4250;4267;4804;4996;
//---------------------------------------------------------
#ifndef LIBOCCA_PCH_HEADER
#define LIBOCCA_PCH_HEADER


#define THIS_IS_READY 0


#if (1)
#define WIN32_LEAN_AND_MEAN
#define _USE_MATH_DEFINES
#define NOMINMAX
#endif

#include <cmath>
#include <algorithm>

#if (1)

#include "occa.hpp"

#else

#include <iostream>
#include <sstream>
#include <map>
#include <vector>

#endif


#endif // LIBOCCA_PCH_HEADER
