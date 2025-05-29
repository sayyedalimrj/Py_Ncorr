// ──────────────────────────────────────────────────────────────────────────────
//  cpp_src/shims/mex.h
//
//  Tiny “dummy” replacement for MATLAB’s <mex.h>.
//  Supplies just enough typedefs, enums and functions so that the
//  original Ncorr C++ code will compile without MATLAB.
//
//  NOTE: **Nothing here actually calls MATLAB** – every API routine
//        either does nothing, returns a default value, or throws a
//        C++ exception (so that logic errors are still caught).
//
//  Last updated: 2025-05-29
// ──────────────────────────────────────────────────────────────────────────────
#ifndef NCORR_DUMMY_MEX_H
#define NCORR_DUMMY_MEX_H

// standard C / C++
#include <cstddef>      // std::size_t
#include <cstdint>
#include <cstdio>
#include <cstdarg> 
#include <cstdlib>
#include <stdexcept>

// ───── Fundamental MATLAB types (minimal) ────────────────────────────────────
using mwSize  = std::size_t;
using mwIndex = std::ptrdiff_t;

// Bare-bones stand-in for MATLAB’s mxArray.
struct mxArray {};

// “Class” IDs – we only need the ones referenced in Ncorr.
enum mxClassID : int {
    mxDOUBLE_CLASS = 6,
    mxINT32_CLASS  = 13,
    mxUINT8_CLASS  = 2,
    mxLOGICAL_CLASS = 1
};

// ───── Dummy creators – always return nullptr ───────────────────────────────
inline mxArray* mxCreateDoubleMatrix(mwSize, mwSize, int)           { return nullptr; }
inline mxArray* mxCreateLogicalMatrix(mwSize, mwSize)               { return nullptr; }
inline mxArray* mxCreateString(const char*)                         { return nullptr; }

// ───── Access helpers – return safe defaults ────────────────────────────────
inline double*   mxGetPr(const mxArray*)              { return nullptr; }
inline bool*     mxGetLogicals(const mxArray*)        { return nullptr; }
inline mwSize    mxGetM(const mxArray*)               { return 0; }
inline mwSize    mxGetN(const mxArray*)               { return 0; }
inline mxArray*  mxGetField(const mxArray*, mwIndex, const char*)   { return nullptr; }
inline mxArray*  mxGetProperty(const mxArray*, mwIndex, const char*){ return nullptr; }

// ───── Type checks (all return false in stub) ───────────────────────────────
inline bool mxIsClass (const mxArray*, const char*)   { return false; }
inline bool mxIsDouble(const mxArray*)                { return false; }
inline bool mxIsLogical(const mxArray*)               { return false; }

// ───── Memory management – no-ops here ───────────────────────────────────────
inline void mxDestroyArray(mxArray*)                  {}
inline void mxFree(void*)                             {}

// ───── MATLAB engine interaction – dummies/throwers ─────────────────────────
inline int  mexCallMATLAB(int, mxArray**, int, mxArray**, const char*)
{ return 0; }   // pretend success

// Simple printf pass-through so the library’s verbose prints still show.
inline int mexPrintf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    int n = std::vfprintf(stderr, fmt, args);
    va_end(args);
    return n;
}

// Error handler – raise C++ exception instead of killing the process.
[[noreturn]] inline void mexErrMsgTxt(const char* msg)
{
    throw std::runtime_error(msg ? msg : "mexErrMsgTxt called");
}

#endif /* NCORR_DUMMY_MEX_H */
