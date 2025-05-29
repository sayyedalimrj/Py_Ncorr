/*─────────────────────────────────────────────────────────────────────────────
  mex.h  –  tiny stand-in so legacy Ncorr C++ code that expected the
            MATLAB MEX API will compile *without* MATLAB.

  Only the handful of symbols referenced inside the Ncorr sources are
  declared.  They do **nothing** at run-time, they merely satisfy the linker.
 ────────────────────────────────────────────────────────────────────────────*/
#ifndef NCORR_DUMMY_MEX_H
#define NCORR_DUMMY_MEX_H

#include <cstddef>      // size_t
#include <cstdint>      // ptrdiff_t
#include <cstdio>       // vfprintf / stderr
#include <cstdarg>      // va_list
#include <stdexcept>    // std::runtime_error

/* ───── basic MATLAB typedefs (opaque) ──────────────────────────────────── */
using mwSize  = std::size_t;
using mwIndex = std::ptrdiff_t;

struct mxArray          { void *data {}; };        // completely opaque
struct mxLogical        { };
constexpr int mxREAL = 0;                          // value is irrelevant

/* ───── dummy creators – return new, empty mxArray so pointer tests pass – */
inline mxArray *mxCreateDoubleMatrix(mwSize, mwSize, int) { return new mxArray; }
inline mxArray *mxCreateLogicalMatrix(mwSize, mwSize)     { return new mxArray; }
inline mxArray *mxCreateString        (const char*)       { return new mxArray; }

/* ───── dummy getters – always return nullptr/0 ─────────────────────────── */
inline double *mxGetPr      (const mxArray*) { return nullptr; }
inline mxLogical *mxGetLogicals(const mxArray*) { return nullptr; }
inline mwSize mxGetM        (const mxArray*) { return 0; }
inline mwSize mxGetN        (const mxArray*) { return 0; }

/* properties / fields – nullptr so code that checks for null will fail safely */
inline mxArray *mxGetProperty(const mxArray*, mwIndex, const char*) { return nullptr; }
inline mxArray *mxGetField   (const mxArray*, mwIndex, const char*) { return nullptr; }

/* ───── destruction – we delete the dummy object ───────────────────────── */
inline void mxDestroyArray(mxArray *p) { delete p; }

/* ───── MEX gateway helpers – all no-ops that pretend success ──────────── */
inline int  mexCallMATLAB(int, mxArray**, int, mxArray**, const char*)
{ return 0; }                                      // pretend “OK”

inline int mexPrintf(const char* fmt, ...)
{
    if (!fmt) return 0;
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    return 0;
}

[[noreturn]] inline void mexErrMsgTxt(const char* msg)
{
    throw std::runtime_error(msg ? msg : "mexErrMsgTxt called");
}

#endif /* NCORR_DUMMY_MEX_H */
