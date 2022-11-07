#define REAL_IS_FLOAT

#ifdef REAL_IS_FLOAT
	typedef float real;
#elif
	typedef double real
#endif