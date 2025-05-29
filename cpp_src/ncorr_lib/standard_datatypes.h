// These are declarations for standard inputs to mex files
  /* ---------- Standard includes required when we are *not* inside a MEX build */
#ifndef NCORR_STDLIB_SAFE_INCLUDES
#define NCORR_STDLIB_SAFE_INCLUDES

// C-headers
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>

// C++-STL
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <memory>

#endif  /* NCORR_STDLIB_SAFE_INCLUDES */

#ifndef STANDARD_DATATYPES_H
#define STANDARD_DATATYPES_H 
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX    // stop windows.h from polluting std::min/max
#  endif
#endif
#include <string> 
#include "matlab_shim.h"

// ----------------------------------------------------------------------//
// This is for a string input -------------------------------------------//
// ----------------------------------------------------------------------//

// NOT THREAD SAFE 
void get_string(std::string &string,const mxArray *mat_buf); // potentially dangerous because user could change the length of the string

// ----------------------------------------------------------------------//
// This is for double scalar input --------------------------------------//
// ----------------------------------------------------------------------//

// NOT THREAD SAFE 
void get_double_scalar(double &scalar,const mxArray *mat_buf);

// ----------------------------------------------------------------------//
// This is for integer scalar input -------------------------------------//
// ----------------------------------------------------------------------//

// NOT THREAD SAFE
void get_integer_scalar(int &scalar,const mxArray *mat_buf);

// ----------------------------------------------------------------------//
// This is for logical scalar input -------------------------------------//
// ----------------------------------------------------------------------//

// NOT THREAD SAFE 
void get_logical_scalar(bool &scalar,const mxArray *mat_buf);

// ----------------------------------------------------------------------//
// This is for a double array input -------------------------------------//
// ----------------------------------------------------------------------//

class class_double_array {
public:
    // Constructor    
    class_double_array();                         // THREAD SAFE
    
    // Properties
    int width;
    int height;
    double *value; 
    
    // Methods
    void reset();                                 // THREAD SAFE
    void alloc(const int &h,const int &w);        // NOT THREAD SAFE
    void free();                                  // NOT THREAD SAFE
}; 

// NOT THREAD SAFE 
void get_double_array(class_double_array &array,const mxArray *mat_buf);

// ----------------------------------------------------------------------//
// This is for a integer array input ------------------------------------//
// ----------------------------------------------------------------------//

class class_integer_array {
public:
    // Constructor    
    class_integer_array();                        // THREAD SAFE
    
    // Properties
    int width;
    int height;
    int *value; 
    
    // Methods
    void reset();                                 // THREAD SAFE
    void alloc(const int &h,const int &w);        // NOT THREAD SAFE
    void free();                                  // NOT THREAD SAFE
};

// NOT THREAD SAFE 
void get_integer_array(class_integer_array &array,const mxArray *mat_buf);

// ----------------------------------------------------------------------//
// This is for a logical array input ------------------------------------//
// ----------------------------------------------------------------------//

class class_logical_array {
public:
    // Constructor    
    class_logical_array();                        // THREAD SAFE
    
    // Properties
    int width;
    int height;
    bool *value; 
    
    // Methods
    void reset();                                 // THREAD SAFE
    void alloc(const int &h,const int &w);        // NOT THREAD SAFE
    void free();                                  // NOT THREAD SAFE
};

// NOT THREAD SAFE
void get_logical_array(class_logical_array &array,const mxArray *mat_buf);

#endif /* STANDARD_DATATYPES_H */
