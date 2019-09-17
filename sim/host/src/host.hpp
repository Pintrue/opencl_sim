#ifndef HOST_HPP
#define HOST_HPP

#include "CL/opencl.h"


#define MAX_32_BITS 4294967295

#define RAD_SCALE_MIN -3.665191429	// -210 degrees
#define RAD_SCALE_MAX 2.443460953	// 140 degrees
#define RAD_SCALE_RANGE (RAD_SCALE_MAX - RAD_SCALE_MIN)

#define INT_RAD_SCALE_MIN 0
#define INT_RAD_SCALE_MAX MAX_32_BITS
#define INT_RAD_SCALE_RANGE (INT_RAD_SCALE_MAX - INT_RAD_SCALE_MIN)
#define INT_TRIG_SCALE_MIN 0
#define INT_TRIG_SCALE_MAX MAX_32_BITS
#define INT_TRIG_SCALE_RANGE (INT_TRIG_SCALE_MAX - INT_TRIG_SCALE_MIN)

#define COMPUTE_UNIT_NUMBER 16
#define NUMBER_OF_ELEMS 6

#define COMPUTE_UNIT_NUMBER_FP 16
#define NUMBER_OF_ELEMS_FP 8

#ifndef M_PI
	#define M_PI 3.14159265359
#endif

#define JNT0_L -M_PI/2
#define JNT0_U M_PI/2
#define JNT1_L 0.0
#define JNT1_U 130.0/180.0*M_PI
#define JNT2_L -M_PI/2
#define JNT2_U 0.0

#define ENABLE_FPKM
// #define ENABLE_KM

#ifdef ENABLE_KM
extern ulong* output_ee_pose;
#endif

#ifdef ENABLE_FPKM
extern double* output_fp_ee_pose;
#endif

void checkStatus(cl_int status, const char* file, int line, const char* msg);
double randAngleRads(double lower, double upper);
uint convertRadsToInt(double radians);
double convertTrigEncToVal(long enc);
bool initOpencl();
void initInput();
void initInput(double jnt_angles[3]);

#ifdef ENABLE_KM
void initKMInput();
void run();
void runKM();
#endif

#ifdef ENABLE_FPKM
void runFPKM();
#endif

void cleanup();

#endif