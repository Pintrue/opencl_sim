#ifndef HOST_HPP
#define HOST_HPP

#include "CL/opencl.h"


// Runtime config
// cl_platform_id platform = NULL;
// cl_device_id device;
// cl_context context = NULL;
// cl_command_queue command_queue;
// cl_program program = NULL;
// cl_kernel kernel;
// cl_kernel km_kernel;

// cl_mem input_jnt_angles_buf;
// cl_mem output_trig_vals_buf;
// cl_mem output_ee_pose_buf;

// // Input data
// uint* input_jnt_angles;
// ulong* output_trig_vals;

// // output data
ulong* output_ee_pose;


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