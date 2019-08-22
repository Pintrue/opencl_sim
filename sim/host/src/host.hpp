#ifndef HOST_HPP
#define HOST_HPP

#include "CL/opencl.h"

extern ulong* output_ee_pose;

void checkStatus(cl_int status, const char* file, int line, const char* msg);
double randAngleRads(double lower, double upper);
uint convertRadsToInt(double radians);
double convertTrigEncToVal(long enc);
bool initOpencl();
void initInput();
void initInput(double jnt_angles[3]);
void initKMInput();
void run();
void runKM();
void cleanup();

#endif