#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctime>
#include <time.h>
#include "CL/opencl.h"
#include "host.hpp"


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

#define NUMBER_OF_ELEMS 6

#ifndef M_PI
	#define M_PI 3.14159265359
#endif

#define JNT0_L -M_PI/2
#define JNT0_U M_PI/2
#define JNT1_L 0.0
#define JNT1_U 130.0/180.0*M_PI
#define JNT2_L -M_PI/2
#define JNT2_U 0.0


using namespace std;


// // Runtime config
cl_platform_id platform = NULL;
cl_device_id device;
cl_context context = NULL;
cl_command_queue command_queue;
cl_program program = NULL;
cl_kernel kernel;
cl_kernel km_kernel;

cl_mem input_jnt_angles_buf;
cl_mem output_trig_vals_buf;
// cl_mem input_trig_vals_buf;
cl_mem output_ee_pose_buf;

// Input data
uint* input_jnt_angles;// = new uint[NUMBER_OF_ELEMS];
ulong* output_trig_vals;// = new ulong[NUMBER_OF_ELEMS];

// long* input_trig_vals;
// ulong* output_ee_pose;


// Function prototypes
// void checkStatus(cl_int status, const char* file, int line, const char* msg);
// double randAngleRads(double lower, double upper);
// uint convertRadsToInt(double radians);
// double convertTrigEncToVal(long enc);
// bool initOpencl();
// void initInput();
// void initInput(double jnt_angles[3]);
// void initKMInput();
// void run();
// void runKM();
// void cleanup();


// Program starts here
// int __main(int argc, char** argv) {
// 	cl_int err;
	
// 	if (!initOpencl()) {
// 		printf("ERROR: Unable to initialize OpenCL at %s: line %d.\n", __FILE__, __LINE__);
// 		return -1;
// 	}

// 	while (1) {
// 		// _ja_0 = -0.502065;
// 		// _ja_1 = -0.675970;
// 		// _ja_2 = -1.911503;

// 		double _jas[3] = {1.068731, 0.894826, -0.340707};
// 		initInput(_jas);
// 		initKMInput();

// 		printf("\nKernel initialization is complete.\n");
// 		printf("Launching the kernel...\n\n");

// 		run();

// 		printf("Output from kernel 1:\n");
// 		printf("------------------------\n");
// 		for (int i = 0; i < NUMBER_OF_ELEMS; ++i) {
// 			// printf("In radians: %u\n", input_jnt_angles[i]);
// 			printf("In encoding: %d: %lu\n", i, output_trig_vals[i]);
// 			// printf("Converted back: %lf\n", convertTrigEncToVal(output_trig_vals[i]));
// 			printf("-----------------------------\n");
// 		}

// 		runKM();

// 		printf("\nOutput from kernel 2:\n");
// 		printf("X = %lu\n", output_ee_pose[0]);
// 		printf("Y = %lu\n", output_ee_pose[1]);
// 		printf("Z = %lu\n", output_ee_pose[2]);

// 		printf("\nTo continue, please enter any key.\n");
// 		char c;
// 		scanf("%c", &c);
// 		printf("\n");
// 	}

// 	printf("Kernel execution complete.\n");

// 	cleanup();

// 	return 0;
// }


void checkStatus(cl_int status,
				const char* file,
				int line,
				const char* msg) {
	if (status != CL_SUCCESS) {
		printf("ERROR: at %s: line %d\n", file, line);
		printf("\t%s.\n", msg);
		exit(status);
	}
}


double randAngleRads(double lower, double upper) {
	return lower + (rand() / (double(RAND_MAX) / (upper - lower)));
}


uint convertRadsToInt(double radians) {
	double encoding = (radians - RAD_SCALE_MIN) / RAD_SCALE_RANGE * INT_RAD_SCALE_RANGE;

	return (uint) round(encoding);
}


double convertTrigEncToVal(long enc) {
	if (enc < INT_TRIG_SCALE_MIN || enc > INT_TRIG_SCALE_MAX) {
		enc = ((enc < INT_TRIG_SCALE_MIN) ? INT_TRIG_SCALE_MIN : enc > INT_TRIG_SCALE_MAX) ? INT_TRIG_SCALE_MAX : enc;
	}

	double val = ((double) enc - INT_TRIG_SCALE_MIN) / INT_TRIG_SCALE_RANGE * 2.0 - 1.0;
	// printf("From encoding %lu to value %lf\n", enc, val);
	return val;
}


bool initOpencl() {
	cl_int err;

	printf("Initializing OpenCL\n");

	// obtain OpenCL platform
	err = clGetPlatformIDs(1, &platform, NULL);
	checkStatus(err, __FILE__, __LINE__, "'clGetPlatformIDs()' failed");

	if (platform == NULL) {
		printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform at %s: line %d\n", __FILE__, __LINE__);
		return false;
	}

	// Obtain the OpenCL device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
	checkStatus(err, __FILE__, __LINE__, "'clGetDeviceIDs()' failed");

	// Create the context for device
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateContext()' failed");

	
	/* Create the program with binary file */

	// - 1. Load the binary file
	// const char* file_name = "bin/hello_world.aocx";
	const char* file_name = "bin/km.aocx";
	FILE* file_ptr;
	file_ptr = fopen(file_name, "rb");
	if (file_ptr == NULL) {
		printf("ERROR: Unable to open binary file %s at %s: line %d\n", file_name, __FILE__, __LINE__);
		return false;
	}

	// - 2. Get file size and allocate that much space for the file
	fseek(file_ptr, 0, SEEK_END);
	size_t binary_length = ftell(file_ptr);
	unsigned char* program_binary = new unsigned char[binary_length];
	rewind(file_ptr);

	// - 3. Read the file into the binary
	if (fread((void*) program_binary, binary_length, 1, file_ptr) == 0) {
		printf("File does not open\n");
		delete[] program_binary;
		fclose(file_ptr);
		return false;
	}

	// - 4. Create the program with the loaded binary file
	program = clCreateProgramWithBinary(context, 1, &device, &binary_length,
				(const unsigned char**) &program_binary, NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateProgramWithBinary()' failed");

	/* END: Create the program with binary file */


	// Build the program
	err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkStatus(err, __FILE__, __LINE__, "'clBuildProgram()' failed");

	// Create the command queue
	command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	checkStatus(err, __FILE__, __LINE__, "clCreateCommandQueue()' failed");

	// Create the kernel
	const char* kernel_name = "cosine_int_32";
	kernel = clCreateKernel(program, kernel_name, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateKernel()' failed");

	const char* km_kernel_name = "get_pose_by_jnts_int_32";
	km_kernel = clCreateKernel(program, km_kernel_name, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateKernel()' failed");

	// Create the input buffer
	input_jnt_angles_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_ELEMS * sizeof(uint), NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'input_jnt_angles_buf' failed");

	// input_trig_vals_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_ELEMS * sizeof(long), NULL, &err);
	// checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'input_trig_vals_buf' failed");

	// Create the output buffer
	output_trig_vals_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUMBER_OF_ELEMS * sizeof(ulong), NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'output_trig_vals_buf' failed");

	output_ee_pose_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 5 * sizeof(ulong), NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'output_ee_pose_buf' failed");

	printf("FINISH INIT.\n");
	return true;
}


void initInput() {
	printf("Entered initInput()\n");

//	double* test = new double[3];
	input_jnt_angles = new uint[NUMBER_OF_ELEMS];
	output_trig_vals = new ulong[NUMBER_OF_ELEMS];
	printf("new\n");	

	// Randomize the input elements
	double ja_0, ja_1, ja_2;		// cosine angle radians
	double _ja_0, _ja_1, _ja_2;		// sine angle radians (offset -pi/2, expressed by cosine)

	// ja_0 = randAngleRads(JNT0_L, JNT0_U); _ja_0 = ja_0 - (M_PI / 2);
	// ja_1 = randAngleRads(JNT1_L, JNT1_U); _ja_1 = ja_1 - (M_PI / 2);
	// ja_2 = randAngleRads(JNT2_L, JNT2_U); _ja_2 = ja_2 - (M_PI / 2);

	ja_0 = 1.068731;
	ja_1 = 0.894826;
	ja_2 = -0.340707;

	uint delta_ja_int[3] = {convertRadsToInt(ja_0), convertRadsToInt(ja_1), convertRadsToInt(ja_2)};
	// _ja_0 = -0.502065;
	// _ja_1 = -0.675970;
	// _ja_2 = -1.911503;
	
	// printf("Before conversion:\n");
	// printf("ja[0] = %lf\n", ja_0);
	// printf("ja[1] = %lf\n", ja_1);
	// printf("ja[2] = %lf\n", ja_2);
	// printf("ja[3] = %lf\n", _ja_0);
	// printf("ja[4] = %lf\n", _ja_1);
	// printf("ja[5] = %lf\n", _ja_2);

	// Convert radians to corresponding integer encoding
	input_jnt_angles[0] = delta_ja_int[0];
	input_jnt_angles[1] = 112855247 + delta_ja_int[1];
	input_jnt_angles[2] = 7877904265 - delta_ja_int[2] - delta_ja_int[1];
	input_jnt_angles[3] = -1104420162 + delta_ja_int[0];
	input_jnt_angles[4] = -991564915 + delta_ja_int[1];
	input_jnt_angles[5] = 6773484103 - delta_ja_int[2] - delta_ja_int[1];
	
	// printf("After conversion:\n");
	// for (int i = 0; i < NUMBER_OF_ELEMS; ++i) {
	// 	printf("ja[%d] = %u\n", i, input_jnt_angles[i]);
	// }
	// TODO: verification output
	printf("Finish init\n");
}


void initInput(double jnt_angles[3]) {
	input_jnt_angles = new uint[NUMBER_OF_ELEMS];
	output_trig_vals = new ulong[NUMBER_OF_ELEMS];

	uint delta_ja_int[3] = {convertRadsToInt(jnt_angles[0]), convertRadsToInt(jnt_angles[1]), convertRadsToInt(jnt_angles[2])};
	// _ja_0 = -0.502065;
	// _ja_1 = -0.675970;
	// _ja_2 = -1.911503;
	
	// printf("Before conversion:\n");
	// printf("ja[0] = %lf\n", ja_0);
	// printf("ja[1] = %lf\n", ja_1);
	// printf("ja[2] = %lf\n", ja_2);
	// printf("ja[3] = %lf\n", _ja_0);
	// printf("ja[4] = %lf\n", _ja_1);
	// printf("ja[5] = %lf\n", _ja_2);

	// Convert radians to corresponding integer encoding
	input_jnt_angles[0] = delta_ja_int[0];
	input_jnt_angles[1] = 112855247 + delta_ja_int[1];
	input_jnt_angles[2] = 7877904265 - delta_ja_int[2] - delta_ja_int[1];
	input_jnt_angles[3] = -1104420162 + delta_ja_int[0];
	input_jnt_angles[4] = -991564915 + delta_ja_int[1];
	input_jnt_angles[5] = 6773484103 - delta_ja_int[2] - delta_ja_int[1];
}


void initKMInput() {
	// input_trig_vals = new long[NUMBER_OF_ELEMS];
	output_ee_pose = new ulong[5];
}


void run() {
	printf("Entered run()\n");
	cl_int err;

	// clock_t begin = clock();
	struct timespec begin, end;
	printf("Start timing\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);

	cl_event kernel_event;
	cl_event finish_event;

	printf("Enqueue writes to input buffers\n");
	// Enqueue write commands to the input buffer
	cl_event write_events[1];
	err = clEnqueueWriteBuffer(command_queue, input_jnt_angles_buf, CL_FALSE,
			0, NUMBER_OF_ELEMS * sizeof(uint), input_jnt_angles, 0, NULL, &write_events[0]);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueWriteBuffer()' for 'input_jnt_angles_buf' failed");

	printf("Setting kernel argument\n");
	// Set kernel argument
	unsigned argi = 0;

	err = clSetKernelArg(kernel, argi++, sizeof(cl_mem),
			&input_jnt_angles_buf);
	checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	err = clSetKernelArg(kernel, argi++, sizeof(cl_mem),
			&output_trig_vals_buf);
	checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	printf("Launching the kernel\n");
	// Launch the kernel
	const size_t global_work_size = NUMBER_OF_ELEMS;
	err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
			&global_work_size, NULL, 1, write_events, &kernel_event);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueNDRangeKernel()' failed");

	// Enqueue read commands on the output buffer
	err = clEnqueueReadBuffer(command_queue, output_trig_vals_buf, CL_FALSE, 0,
			NUMBER_OF_ELEMS * sizeof(ulong), output_trig_vals, 1, &kernel_event,
			&finish_event);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueReadBuffer()' failed");

	// Release local event
	clReleaseEvent(write_events[0]);

	// Wait for the output write to finish
	clWaitForEvents(1, &finish_event);


	// clock_t end = clock();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	double elapsed_time = (end.tv_sec - begin.tv_sec) * 1e6 + (end.tv_nsec - begin.tv_nsec) / 1e3;

	// double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
	printf("Kernel time: %0.3lf microseconds.\n", elapsed_time);

	// Release all events
	clReleaseEvent(kernel_event);
	clReleaseEvent(finish_event);
	printf("Finish kernel run()\n");
}


void runKM() {
	cl_int err;

	struct timespec begin, end;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);

	cl_event kernel_event;
	cl_event finish_event;

	// Enqueue write commands to the input buffer
	// cl_event write_events[1];
	// err = clEnqueueWriteBuffer(command_queue, output_trig_vals_buf, CL_FALSE,
	// 		0, NUMBER_OF_ELEMS * sizeof(long), input_trig_vals, 0, NULL, &write_events[0]);
	// checkStatus(err, __FILE__, __LINE__, "'clEnqueueWriteBuffer()' for 'output_trig_vals_buf' failed");

	// err = clEnqueueWriteBuffer(command_queue, input_trig_vals_buf, CL_FALSE,
	// 		0, NUMBER_OF_ELEMS * sizeof(long), output_trig_vals, 0, NULL, &write_events[0]);
	// checkStatus(err, __FILE__, __LINE__, "'clEnqueueWriteBuffer()' for 'input_trig_vals_buf' failed");


	// Set kernel argument
	unsigned argi = 0;

	err = clSetKernelArg(km_kernel, argi++, sizeof(cl_mem),
			&output_trig_vals_buf);
	checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	err = clSetKernelArg(km_kernel, argi++, sizeof(cl_mem),
			&output_ee_pose_buf);
	checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	// Launch the kernel
	const size_t global_work_size = 1;
	err = clEnqueueNDRangeKernel(command_queue, km_kernel, 1, NULL,
			&global_work_size, NULL, 0, NULL, &kernel_event);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueNDRangeKernel()' failed");

	// Enqueue read commands on the output buffer
	err = clEnqueueReadBuffer(command_queue, output_ee_pose_buf, CL_FALSE, 0,
			5 * sizeof(ulong), output_ee_pose, 1, &kernel_event,
			&finish_event);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueReadBuffer()' failed");

	// Release local event
	// clReleaseEvent(write_events[0]);

	// Wait for the output write to finish
	clWaitForEvents(1, &finish_event);


	// clock_t end = clock();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	double elapsed_time = (end.tv_sec - begin.tv_sec) * 1e6 + (end.tv_nsec - begin.tv_nsec) / 1e3;

	// double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
	printf("Kernel time: %0.3lf microseconds.\n", elapsed_time);

	// Release all events
	clReleaseEvent(kernel_event);
	clReleaseEvent(finish_event);
}


void cleanup() {
	if (kernel) {
		clReleaseKernel(kernel);
	}

	if (km_kernel) {
		clReleaseKernel(km_kernel);
	}

	if (program) {
		clReleaseProgram(program);
	}

	if (command_queue) {
		clReleaseCommandQueue(command_queue);
	}

	if (context) {
		clReleaseContext(context);
	}

	if (input_jnt_angles_buf) {
		clReleaseMemObject(input_jnt_angles_buf);
	}

	// if (input_trig_vals_buf) {
	// 	clReleaseMemObject(input_trig_vals_buf);
	// }

	if (output_trig_vals_buf) {
		clReleaseMemObject(output_trig_vals_buf);
	}

	if (output_ee_pose_buf) {
		clReleaseMemObject(output_ee_pose_buf);
	}

	if (input_jnt_angles) {
		delete[] input_jnt_angles;
	}

	if (output_trig_vals) {
		delete[] output_trig_vals;
	}

	if (output_ee_pose) {
		delete[] output_ee_pose;
	}

	printf("FINISH CLEANUP.\n");
}
