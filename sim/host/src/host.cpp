#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "CL/opencl.h"
// #include "AOCLUtils/aocl_utils.h"


#define RAD_SCALE_MIN -3.665191429	// -210 degrees
#define RAD_SCALE_MAX 2.443460953	// 140 degrees
#define RAD_SCALE_RANGE (RAD_SCALE_MAX - RAD_SCALE_MIN)

#define INT_RAD_SCALE_RANGE 4294967295	// 32-bit encoding

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

static const int thread_id_to_output = 2;

// Runtime config
cl_platform_id platform = NULL;
cl_device_id device;
cl_context context = NULL;
cl_command_queue command_queue;
cl_program program = NULL;
cl_kernel kernel;

cl_mem input_jnt_angles_buf;
cl_mem output_trig_vals_buf;

// Input data
uint* input_jnt_angles;
long* output_trig_vals;


// Function prototypes
void checkStatus(cl_int status, const char* file, int line, const char* msg);
float randAngleRads(float lower, float upper);
uint convertRadsToInt(float radians);
bool initOpencl();
void initInput();
void run();
void cleanup();


// Program starts here
int main(int argc, char** argv) {
	cl_int err;
	
	if (!initOpencl()) {
		printf("ERROR: Unable to initialize OpenCL at %s: line %d.\n", __FILE__, __LINE__);
		return -1;
	}

	initInput();

	err = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &thread_id_to_output);
	checkStatus(err, __FILE__, __LINE__, "Failed to set kernel arg 0");

	printf("\nKernel initialization is complete.\n");
	printf("Launching the kernel...\n\n");

	// Create work-item set
	size_t local_size[3] = {8, 1, 1};
	size_t global_size[3] = {8, 1, 1};

	// launch the kernel
	err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
	checkStatus(err, __FILE__, __LINE__, "Launching NDKernel failed");

	err = clFinish(command_queue);
	checkStatus(err, __FILE__, __LINE__, "Does not finish smoothly");

	printf("Kernel execution complete.\n");

	cleanup();

	return 0;
}


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
	// double encoding = (radians - RAD_SCALE_MIN) / RAD_SCALE_RANGE * INT_RAD_SCALE_RANGE;
	// return (uint) round(encoding);
	return (uint) ((radians - RAD_SCALE_MIN) / RAD_SCALE_RANGE * INT_RAD_SCALE_RANGE);
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
	const char* file_name = "bin/hello_world.aocx";
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
	const char* kernel_name = "hello_world";
	kernel = clCreateKernel(program, kernel_name, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateKernel()' failed");

	// Create the input buffer
	input_jnt_angles_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, NUMBER_OF_ELEMS * sizeof(uint), NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'input_jnt_angles_buf' failed");

	// Create the output buffer
	output_trig_vals_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, NUMBER_OF_ELEMS * sizeof(long), NULL, &err);
	checkStatus(err, __FILE__, __LINE__, "'clCreateBuffer()' for 'output_trig_vals_buf' failed");


	printf("FINISH INIT.\n");
	return true;
}


void initInput() {
	input_jnt_angles = new uint[NUMBER_OF_ELEMS];
	output_trig_vals = new long[NUMBER_OF_ELEMS];

	// Randomize the input elements
	double ja_0, ja_1, ja_2;		// cosine angle radians
	double _ja_0, _ja_1, _ja_2;	// sine angle radians (offset -pi/2, expressed by cosine)

	ja_0 = randAngleRads(JNT0_L, JNT0_U); _ja_0 = ja_0 - (M_PI / 2);
	ja_1 = randAngleRads(JNT1_L, JNT1_U); _ja_1 = ja_1 - (M_PI / 2);
	ja_2 = randAngleRads(JNT2_L, JNT2_U); _ja_2 = ja_2 - (M_PI / 2);
	
	printf("Before conversion:\n");
	printf("ja[0] = %lf\n", ja_0);
	printf("ja[1] = %lf\n", ja_1);
	printf("ja[2] = %lf\n", ja_2);
	printf("ja[3] = %lf\n", _ja_0);
	printf("ja[4] = %lf\n", _ja_1);
	printf("ja[5] = %lf\n", _ja_2);

	// Convert radians to corresponding integer encoding
	input_jnt_angles[0] = convertRadsToInt(ja_0);
	input_jnt_angles[1] = convertRadsToInt(ja_1);
	input_jnt_angles[2] = convertRadsToInt(ja_2);
	input_jnt_angles[3] = convertRadsToInt(_ja_0);
	input_jnt_angles[4] = convertRadsToInt(_ja_1);
	input_jnt_angles[5] = convertRadsToInt(_ja_2);
	
	printf("After conversion:\n");
	for (int i = 0; i < NUMBER_OF_ELEMS; ++i) {
		printf("ja[%d] = %u\n", i, input_jnt_angles[i]);
	}
	// TODO: verification output
}


void run() {

}


void cleanup() {
	if (kernel) {
		clReleaseKernel(kernel);
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

	if (output_trig_vals_buf) {
		clReleaseMemObject(output_trig_vals_buf);
	}

	if (input_jnt_angles) {
		delete[] input_jnt_angles;
	}

	if (output_trig_vals) {
		delete[] output_trig_vals;
	}


	printf("FINISH CLEANUP.\n");
}
