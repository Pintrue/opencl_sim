#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define vec

__kernel void cosine_int_32(__global const uint* restrict jnt_angles,
							__global ulong* restrict output) {

	__local char grad_table_32[4095];
	__local long intercept_table_32[4095];

	// init the LUT of integer-encoded cosine function
	#pragma unroll 16
	for (uint i = 0; i < 351; ++i) {
		grad_table_32[i] = -1;
		intercept_table_32[i] = 287708254;
	}

	#pragma unroll 16
	for (uint i = 351; i < 819; ++i) {
		grad_table_32[i] = 1;
		intercept_table_32[i] = -368140054;
	}

	#pragma unroll 16
	for (uint i = 819; i < 1872; ++i) {
		grad_table_32[i] = 3;
		intercept_table_32[i] = -2262381335;
	}

	#pragma unroll 16
	for (uint i = 1872; i < 2457; ++i) {
		grad_table_32[i] = 1;
		intercept_table_32[i] = 1717986918;
	}

	#pragma unroll 16
	for (uint i = 2457; i < 3042; ++i) {
		grad_table_32[i] = -1;
		intercept_table_32[i] = 6871947672;
	}

	#pragma unroll 16
	for (uint i = 3042; i < 4096; ++i) {
		grad_table_32[i] = -3;
		intercept_table_32[i] = 13099500927;
	}
	//end init

	// obtain work-item index and then the angle at that index
	int idx = get_global_id(0);
	uint angle_input = jnt_angles[idx];
	
	// mask and shift to obtain LUT index
	uint angle_idx = (angle_input & 0xFFF00000) >> 20;

	// obtain trigonometry encoding value at that LUT index
	output[idx] = (ulong) grad_table_32[angle_idx] * angle_input + intercept_table_32[angle_idx];
	printf("rad = %u, output[%d] = %lu\n", angle_input, idx, output[idx]);
}


__kernel void get_pose_by_jnts_int_32(__global const long* restrict trig_vals,
										__global ulong* restrict ee_pose) {
	// __local long link_lengths[4];

	// Link lengths in integer-encoding
	// 290 = 290;	// base height
	// 524 = 524;
	// 1064 = 1064;
	// 1687 = 1687;

	long d2 = 290;
	long d3 = 524 * 3581808896;		// l1 * sin(a2)
	long d4 = 1064 * trig_vals[4];	// l2 * sin(a3)
	long d5 = 1687 * trig_vals[5];	// l3 * sin(a4)

	// Aggregate the four sections above to obtain Y-coordinate
	ee_pose[1] = d2 + d3 + d4 + d5;

	long d6 = 1687 * trig_vals[2];	// l3 * cos(a4)
	long d7 = 1064 * trig_vals[1];	// l2 * cos(a3)
	long d8 = 524 * 3745731782; 	// l1 * cos(a2)

	long d1 = d6 - d7 + d8;

	// Use base angle to obtain X- and Z-coordinates
	ee_pose[0] = d1 * trig_vals[3];	// d1 * sin(a1)
	ee_pose[2] = d1 * trig_vals[0];	// d1 * cos(a1)

	ee_pose[3] = d1;
	ee_pose[4] = trig_vals[0];
	ee_pose[5] = trig_vals[3];
	printf("x = %lu, y = %lu, z = %lu, d1 = %lu, cos(a1) = %lu, sin(a1) = %lu\n", ee_pose[0], ee_pose[1], ee_pose[2], ee_pose[3], ee_pose[4], ee_pose[5]);
}


__kernel void get_pose_by_jnts(__global const double* restrict radians,
								__global double* restrict ee_pose) {
	// #ifdef vec
	// printf("Y\n");
	// double8 radians_vec = (double8) (radians[0], radians[1], radians[2],
	// 								radians[3], radians[4], radians[5],
	// 								0.0, 0.0); 
	
	// double8 trig_vals_vec = cos(radians_vec);
	// // printf("Output from floating-point version\n");

	// // printf("trig_vals_vec[0] = %lf\n", trig_vals_vec.s0);
	// // printf("trig_vals_vec[1] = %lf\n", trig_vals_vec.s1);
	// // printf("trig_vals_vec[2] = %lf\n", trig_vals_vec.s2);
	// // printf("trig_vals_vec[3] = %lf\n", trig_vals_vec.s3);
	// // printf("trig_vals_vec[4] = %lf\n", trig_vals_vec.s4);
	// // printf("trig_vals_vec[5] = %lf\n", trig_vals_vec.s5);

	// #else
	// printf("N\n");
	// double init_jnt_angles[2] = {atan2(1.7, 10.5), atan2(3.5, 16.5)};

	// double inter_angles[4];

	// double a2 = atan2(3.5, 3.9);
	// inter_angles[0] = radians[0];
	// inter_angles[1] = atan2(3.5, 3.90);
	// inter_angles[2] = atan2(1.7, 10.50) + radians[1];
	// inter_angles[3] = atan2(3.5, 16.50) - radians[2] - radians[1];

	// double trig_vals[6];

	// #pragma unroll 6
	// for (int i = 0; i < 6; ++i) {
	// 	trig_vals[i] = cos(radians[i]);
	// }

	double link_lengths[3] = {sqrt(3.5*3.5+3.9*3.9), sqrt(1.7*1.7+10.5*10.5), sqrt(3.5*3.5+16.5*16.5)};
	
	// y = base_height/2.9;
	ee_pose[1] = 2.9;

	// y += l1*sin(a2) + l2*sin(a3) + l3*sin(a4);
	#pragma unroll
	for (int i = 0; i < 3; ++i) {
		double temp = link_lengths[i] * cos(radians[i + 5]);
		printf("a%d: %lf, sin(a%d): %lf\n", i + 2, radians[i + 5], i + 2, cos(radians[i + 5]));
		ee_pose[1] += temp;
	}
	
	// d1 = -l2*cos(a3);
	double d1 = -link_lengths[1] * cos(radians[2]);

	// d1 += l1*cos(a2)+l3*cos(a4);
	#pragma unroll
	for (int i = 0; i < 3; i += 2) {
		d1 += link_lengths[i] * cos(radians[i + 1]);
	}
	printf("d1 = %lf\n", d1);

	ee_pose[0] = d1 * cos(radians[4]);
	ee_pose[2] = d1 * cos(radians[0]);
	printf("x = %lf, y = %lf, z = %lf\n", ee_pose[0], ee_pose[1], ee_pose[2]);
}
