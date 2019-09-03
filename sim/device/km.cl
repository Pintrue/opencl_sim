// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#define vec

#pragma OPENCL EXTENSION cl_intel_channels : enable
// channel long trig_val_chan __attribute__((depth(6)));

#define CU_NUM 4
#define NUM_JA_PER_SET 6
channel long all_trig_val_chnls[CU_NUM] __attribute__((depth(NUM_JA_PER_SET)));


__kernel void cosine_int_32(__global const uint* restrict jnt_angles) {

	__local char grad_table_32[4096];
	__local long intercept_table_32[4096];

	// init the LUT of integer-encoded cosine function
	#pragma unroll 2
	for (uint i = 0; i < 351; ++i) {
		grad_table_32[i] = -1;
		intercept_table_32[i] = 287708254;
	}

	#pragma unroll 4
	for (uint i = 351; i < 819; ++i) {
		grad_table_32[i] = 1;
		intercept_table_32[i] = -368140054;
	}

	#pragma unroll 4
	for (uint i = 819; i < 1872; ++i) {
		grad_table_32[i] = 3;
		intercept_table_32[i] = -2262381335;
	}

	#pragma unroll 4
	for (uint i = 1872; i < 2457; ++i) {
		grad_table_32[i] = 1;
		intercept_table_32[i] = 1717986918;
	}

	#pragma unroll 4
	for (uint i = 2457; i < 3042; ++i) {
		grad_table_32[i] = -1;
		intercept_table_32[i] = 6871947672;
	}

	#pragma unroll 8
	for (uint i = 3042; i < 4096; ++i) {
		grad_table_32[i] = -3;
		intercept_table_32[i] = 13099500927;
	}
	//end init

	// obtain work-item index and then the angle at that index
	// int idx = get_global_id(0);

	for (uint cu_idx = 0; cu_idx < CU_NUM; ++cu_idx) {
		for (uint idx = 0; idx < NUM_JA_PER_SET; ++idx) {
			uint angle_input = jnt_angles[cu_idx * CU_NUM + idx];

			// mask and shift to obtain LUT index
			uint angle_idx = (angle_input & 0xFFF00000) >> 20;

			// obtain trigonometry encoding value at that LUT index
			ulong trig_val_temp = (ulong) grad_table_32[angle_idx] * angle_input + intercept_table_32[angle_idx];

			// write_channel_intel(trig_val_chan, trig_val_temp);
			write_channel_intel(all_trig_val_chnls[cu_idx], trig_val_temp);
		}
	}
	// printf("rad = %u, output[%d] = %lu\n", angle_input, idx, output[idx]);
}


__attribute__((num_compute_units(CU_NUM)))

// aggregate all the trig. values from the block-read channel before moving on
__kernel void get_pose_by_jnts_int_32(__global ulong* restrict ee_pose) {
	long trig_vals_channeled[NUM_JA_PER_SET];
	// int cu_idx = get_compute_id(0);
	int cu_idx = get_global_id(0);

	for (int i = 0; i < NUM_JA_PER_SET; ++i) {
		// trig_vals_channeled[i] = read_channel_intel(trig_val_chan);
		trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[cu_idx]);
	}

	int offset = cu_idx * CU_NUM;

	long d2 = 290;
	long d3 = 524 * 3581808896;					// l1 * sin(a2)
	long d4 = 1064 * trig_vals_channeled[4];	// l2 * sin(a3)
	long d5 = 1687 * trig_vals_channeled[5];	// l3 * sin(a4)

	// Aggregate the four sections above to obtain Y-coordinate
	ee_pose[offset + 1] = d2 + d3 + d4 + d5;

	long d6 = 1687 * trig_vals_channeled[2];	// l3 * cos(a4)
	long d7 = 1064 * trig_vals_channeled[1];	// l2 * cos(a3)
	long d8 = 524 * 3745731782; 				// l1 * cos(a2)

	long d1 = d6 - d7 + d8;

	// Use base angle to obtain X- and Z-coordinates
	ee_pose[offset + 0] = d1 * trig_vals_channeled[3];	// d1 * sin(a1)
	ee_pose[offset + 2] = d1 * trig_vals_channeled[0];	// d1 * cos(a1)

	ee_pose[offset + 3] = d1;
	ee_pose[offset + 4] = trig_vals_channeled[0];
	ee_pose[offset + 5] = trig_vals_channeled[3];
}


__kernel void get_pose_by_jnts(__global const double* restrict radians,
								__global double* restrict ee_pose) {
	double link_lengths[3] = {sqrt(3.5*3.5+3.9*3.9), sqrt(1.7*1.7+10.5*10.5), sqrt(3.5*3.5+16.5*16.5)};
	
	// y = base_height/2.9;
	ee_pose[1] = 2.9;

	// y += l1*sin(a2) + l2*sin(a3) + l3*sin(a4);
	#pragma unroll
	for (int i = 0; i < 3; ++i) {
		ee_pose[1] += link_lengths[i] * cos(radians[i + 5]);
	}
	
	// d1 = -l2*cos(a3);
	double d1 = -link_lengths[1] * cos(radians[2]);

	// d1 += l1*cos(a2)+l3*cos(a4);
	#pragma unroll
	for (int i = 0; i < 3; i += 2) {
		d1 += link_lengths[i] * cos(radians[i + 1]);
	}

	ee_pose[0] = d1 * cos(radians[4]);
	ee_pose[2] = d1 * cos(radians[0]);
}
