#include <time.h>
#include <stdio.h>

double rtclock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Print elapsed time in milliseconds to stderr (parsed by run_convbench.py) */
void printTime(double t_seconds) {
    fprintf(stderr, "%.6f ms\n", t_seconds * 1000.0);
}
