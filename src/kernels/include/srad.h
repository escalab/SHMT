#define STR_SIZE 256

#ifdef RD_WG_SIZE_0_0
        #define SRAD_BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define SRAD_BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define SRAD_BLOCK_SIZE RD_WG_SIZE
#else
        #define SRAD_BLOCK_SIZE 16
#endif

#define CPU
#define TIMER
#define OUTPUT

