#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <ctime>
#include <time.h>
#include <vector>
#include "CL/opencl.h"
#include "host.hpp"


// #define DEBUG_PRTOUT

using namespace std;


// // Runtime config
cl_platform_id platform = NULL;
cl_device_id device;
cl_context context = NULL;
// cl_command_queue command_queue;
size_t cq_size = 3;
vector<cl_command_queue> command_queues(cq_size);
cl_program program = NULL;

#ifdef ENABLE_KM
cl_kernel kernel;
cl_kernel km_kernel;

cl_mem input_jnt_angles_buf;
// cl_mem output_trig_vals_buf;
// cl_mem input_trig_vals_buf;
cl_mem output_ee_pose_buf;

// Input data
uint* input_jnt_angles;

ulong* output_ee_pose;
#endif

#ifdef ENABLE_FPKM
cl_kernel fp_km_kernel;

cl_mem input_radians_buf;
cl_mem output_fp_ee_pose_buf;

double* input_radians;
double* output_fp_ee_pose;
#endif

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
		printf("ERROR (code %d): at %s: line %d\n", status, file, line);
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
	#ifdef DEBUG_PRTOUT
	cl_device_id* devices;
	devices = (cl_device_id*) malloc(sizeof(cl_device_id) * 5);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 5, devices, NULL);

	size_t value_size;
	char* value;
	for (int i = 0; i < 5; ++i) {
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &value_size);
		value = (char*) malloc(value_size);
		clGetDeviceInfo(devices[i], CL_DEVICE_NAME, value_size, value, NULL);
		printf("%d. Device: %s\n", i + 1, value);
		free(value);
	}
	#endif

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
	// command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	// checkStatus(err, __FILE__, __LINE__, "clCreateCommandQueue()' failed");
	for (int i = 0; i < cq_size; ++i) {
		command_queues[i] = clCreateCommandQueue(context, device,
								CL_QUEUE_PROFILING_ENABLE, &err);
	}

	// Create the kernel
	#ifdef ENABLE_KM
	const char* kernel_name = "cosine_int_32";
	kernel = clCreateKernel(program, kernel_name, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateKernel()' failed");

	const char* km_kernel_name = "get_pose_by_jnts_int_32";
	km_kernel = clCreateKernel(program, km_kernel_name, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateKernel()' failed");
	#endif

	#ifdef ENABLE_FPKM
	const char* fp_km_kernel_name = "get_pose_by_jnts";
	fp_km_kernel = clCreateKernel(program, fp_km_kernel_name, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateKernel()' failed");
	#endif

	// Create the input buffer
	#ifdef ENABLE_KM
	input_jnt_angles_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_ELEMS * sizeof(uint) * COMPUTE_UNIT_NUMBER, NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'input_jnt_angles_buf' failed");

	output_ee_pose_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 6 * sizeof(ulong) * COMPUTE_UNIT_NUMBER, NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'output_ee_pose_buf' failed");
	#endif

	#ifdef ENABLE_FPKM
	input_radians_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_ELEMS_FP * sizeof(double) * COMPUTE_UNIT_NUMBER_FP, NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'input_radians_buf' failed");

	output_fp_ee_pose_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 3 * sizeof(double) * COMPUTE_UNIT_NUMBER_FP, NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'output_fp_ee_pose_buf' failed");
	#endif

	printf("FINISH INIT.\n");
	return true;
}


void initInput() {
	#ifdef ENABLE_KM
	input_jnt_angles = new uint[NUMBER_OF_ELEMS * COMPUTE_UNIT_NUMBER];
	#endif

	#ifdef ENABLE_FPKM
	input_radians = new double[NUMBER_OF_ELEMS_FP * COMPUTE_UNIT_NUMBER_FP];
	output_fp_ee_pose = new double[3 * COMPUTE_UNIT_NUMBER_FP];
	#endif

	// Randomize the input elements
	double ja_0, ja_1, ja_2;		// cosine angle radians
	double _ja_0, _ja_1, _ja_2;		// sine angle radians (offset -pi/2, expressed by cosine)

	// ja_0 = randAngleRads(JNT0_L, JNT0_U); _ja_0 = ja_0 - (M_PI / 2);
	// ja_1 = randAngleRads(JNT1_L, JNT1_U); _ja_1 = ja_1 - (M_PI / 2);
	// ja_2 = randAngleRads(JNT2_L, JNT2_U); _ja_2 = ja_2 - (M_PI / 2);

	ja_0 = 1.068731;
	ja_1 = 0.894826;
	ja_2 = -0.340707;

	#ifdef ENABLE_KM
	uint delta_ja_int[3] = {convertRadsToInt(ja_0), convertRadsToInt(ja_1), convertRadsToInt(ja_2)};

	for (int cu_idx = 0; cu_idx < COMPUTE_UNIT_NUMBER; ++cu_idx) {
		int offset = cu_idx * NUMBER_OF_ELEMS;
		input_jnt_angles[offset + 0] = delta_ja_int[0];
		input_jnt_angles[offset + 1] = 112855247 + delta_ja_int[1];
		input_jnt_angles[offset + 2] = 7877904265 - delta_ja_int[2] - delta_ja_int[1];
		input_jnt_angles[offset + 3] = -1104420162 + delta_ja_int[0];
		input_jnt_angles[offset + 4] = -991564915 + delta_ja_int[1];
		input_jnt_angles[offset + 5] = 6773484103 - delta_ja_int[2] - delta_ja_int[1];
	}
	#endif

	#ifdef ENABLE_FPKM
	for (int cu_idx = 0; cu_idx < COMPUTE_UNIT_NUMBER_FP; ++cu_idx) {
		int offset = cu_idx * NUMBER_OF_ELEMS_FP;
		input_radians[offset + 0] = ja_0;
		input_radians[offset + 1] = atan2(3.5, 3.9);
		input_radians[offset + 2] = atan2(1.70, 10.50) + ja_1;
		input_radians[offset + 3] = atan2(3.50, 16.50) - ja_2 - ja_1;
		input_radians[offset + 4] = ja_0 - M_PI_2;
		input_radians[offset + 5] = input_radians[offset + 1] - M_PI_2;
		input_radians[offset + 6] = input_radians[offset + 2] - M_PI_2;
		input_radians[offset + 7] = input_radians[offset + 3] - M_PI_2;
	}
	#endif

	printf("Finish init of input joint angles\n");
}


void initInput(double jnt_angles[3]) {
	// TODO:: CHANGE THIS FUNCTION!!
	input_jnt_angles = new uint[NUMBER_OF_ELEMS];
	// output_trig_vals = new ulong[NUMBER_OF_ELEMS];

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


#ifdef ENABLE_KM
void initKMInput() {
	output_ee_pose = new ulong[6 * COMPUTE_UNIT_NUMBER];
}


void run() {
	cl_int err;

	// clock_t begin = clock();
	struct timespec begin, end;
	printf("Start timing\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);

	cl_event kernel_event;
	// cl_event finish_event;

	// Enqueue write commands to the input buffer
	cl_event write_events[1];
	err = clEnqueueWriteBuffer(command_queues[0], input_jnt_angles_buf, CL_FALSE,
			0, NUMBER_OF_ELEMS * sizeof(uint) * COMPUTE_UNIT_NUMBER, input_jnt_angles, 0, NULL, &write_events[0]);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueWriteBuffer()' for 'input_jnt_angles_buf' failed");

	// Set kernel argument
	unsigned argi = 0;

	err = clSetKernelArg(kernel, argi++, sizeof(cl_mem),
			&input_jnt_angles_buf);
	checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	// err = clSetKernelArg(kernel, argi++, sizeof(cl_mem),
	// 		&output_trig_vals_buf);
	// checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	// Launch the kernel
	const size_t global_work_size = 1;
	err = clEnqueueNDRangeKernel(command_queues[0], kernel, 1, NULL,
			&global_work_size, NULL, 1, write_events, &kernel_event);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueNDRangeKernel()' failed");

	// Enqueue read commands on the output buffer
	// err = clEnqueueReadBuffer(command_queues[0], output_trig_vals_buf, CL_FALSE, 0,
	// 		NUMBER_OF_ELEMS * sizeof(ulong), output_trig_vals, 1, &kernel_event,
	// 		&finish_event);
	// checkStatus(err, __FILE__, __LINE__, "'clEnqueueReadBuffer()' failed");

	// Release local event
	clReleaseEvent(write_events[0]);

	// Wait for the output write to finish
	clWaitForEvents(1, &kernel_event);
	// clWaitForEvents(1, &finish_event);


	// clock_t end = clock();
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	double elapsed_time = (end.tv_sec - begin.tv_sec) * 1e6 + (end.tv_nsec - begin.tv_nsec) / 1e3;

	// double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;
	printf("Kernel 1 time: %0.3lf microseconds.\n", elapsed_time);

	// Release all events
	clReleaseEvent(kernel_event);
	// clReleaseEvent(finish_event);
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

	// err = clSetKernelArg(km_kernel, argi++, sizeof(cl_mem),
	// 		&output_trig_vals_buf);
	// checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	err = clSetKernelArg(km_kernel, argi++, sizeof(cl_mem),
			&output_ee_pose_buf);
	checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	// Launch the kernel
	const size_t global_work_size = COMPUTE_UNIT_NUMBER;
	err = clEnqueueNDRangeKernel(command_queues[1], km_kernel, 1, NULL,
			&global_work_size, NULL, 0, NULL, &kernel_event);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueNDRangeKernel()' failed");

	// Enqueue read commands on the output buffer
	err = clEnqueueReadBuffer(command_queues[1], output_ee_pose_buf, CL_FALSE, 0,
			6 * sizeof(ulong) * COMPUTE_UNIT_NUMBER, output_ee_pose, 1, &kernel_event,
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
	printf("Kernel time 2: %0.3lf microseconds.\n", elapsed_time);

	// Release all events
	clReleaseEvent(kernel_event);
	clReleaseEvent(finish_event);
}
#endif


#ifdef ENABLE_FPKM
void runFPKM() {
	cl_int err;

	// clock_t begin = clock();
	struct timespec begin, end;
	printf("Start timing\n");
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &begin);

	cl_event kernel_event;
	cl_event finish_event;

	// Enqueue write commands to the input buffer
	cl_event write_events[1];
	err = clEnqueueWriteBuffer(command_queues[2], input_radians_buf, CL_FALSE,
			0, NUMBER_OF_ELEMS_FP * sizeof(double) * COMPUTE_UNIT_NUMBER_FP, input_radians, 0, NULL, &write_events[0]);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueWriteBuffer()' for 'input_radians_buf' failed");

	// Set kernel argument
	unsigned argi = 0;

	err = clSetKernelArg(fp_km_kernel, argi++, sizeof(cl_mem),
			&input_radians_buf);
	checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	err = clSetKernelArg(fp_km_kernel, argi++, sizeof(cl_mem),
			&output_fp_ee_pose_buf);
	checkStatus(err, __FILE__, __LINE__, "'clSetKernelArg()' failed");

	// Launch the kernel
	const size_t global_work_size = COMPUTE_UNIT_NUMBER_FP;
	const size_t local_work_size = 1;
	err = clEnqueueNDRangeKernel(command_queues[2], fp_km_kernel, 1, NULL,
			&global_work_size, &local_work_size, 1, write_events, &kernel_event);
	checkStatus(err, __FILE__, __LINE__, "'clEnqueueNDRangeKernel()' failed");

	// Enqueue read commands on the output buffer
	err = clEnqueueReadBuffer(command_queues[2], output_fp_ee_pose_buf, CL_FALSE, 0,
			3 * sizeof(double) * COMPUTE_UNIT_NUMBER_FP, output_fp_ee_pose, 1, &kernel_event,
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
	printf("Kernel time 3: %0.3lf microseconds.\n", elapsed_time);

	// Release all events
	clReleaseEvent(kernel_event);
	clReleaseEvent(finish_event);
}
#endif


void cleanup() {
	#ifdef ENABLE_KM
	if (kernel) {
		clReleaseKernel(kernel);
	}

	if (km_kernel) {
		clReleaseKernel(km_kernel);
	}
	#endif

	#ifdef ENABLE_FPKM
	if (fp_km_kernel) {
		clReleaseKernel(fp_km_kernel);
	}
	#endif

	if (program) {
		clReleaseProgram(program);
	}

	for (int i = 0; i < cq_size; ++i) {
		if (command_queues[i]) {
			clReleaseCommandQueue(command_queues[i]);
		}
	}

	if (context) {
		clReleaseContext(context);
	}

	#ifdef ENABLE_KM
	if (input_jnt_angles_buf) {
		clReleaseMemObject(input_jnt_angles_buf);
	}

	if (output_ee_pose_buf) {
		clReleaseMemObject(output_ee_pose_buf);
	}
	
	if (input_jnt_angles) {
		delete[] input_jnt_angles;
	}

	if (output_ee_pose) {
		delete[] output_ee_pose;
	}
	#endif
	
	#ifdef ENABLE_FPKM
	if (input_radians_buf) {
		clReleaseMemObject(input_radians_buf);
	}

	if (output_fp_ee_pose_buf) {
		clReleaseMemObject(output_fp_ee_pose_buf);
	}


	if (input_radians) {
		delete[] input_radians;
	}

	if (output_fp_ee_pose) {
		delete[] output_fp_ee_pose;
	}
	#endif

	printf("FINISH CLEANUP.\n");
}
