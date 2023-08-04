#ifndef __PERFORMANCE_H__
#define __PERFORMANCE_H__

class TimeBreakDown{
public:
    /* useful APIs */
    
    /* Get the total latency of a kernel call. 
       The actual kernel execution latency is averaged over 
       "iter" times for better consistent result. */
    double get_total_time_ms(int iter);
    
    // section timing
    double input_time_ms;
    double kernel_time_ms;
    double output_time_ms;
};
#endif

