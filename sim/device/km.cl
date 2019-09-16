// #pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#define vec

#pragma OPENCL EXTENSION cl_intel_channels : enable
// channel long trig_val_chan __attribute__((depth(6)));

#define CU_NUM 16
#define FP_SIMUL_SET 8
#define NUM_JA_PER_SET 6
#define NUM_OUT_POSE_PER_SET 6
#define NUM_RAD_PER_SET 8
#define NUM_OUT_POSE_PER_SET_FP 3


channel ulong all_trig_val_chnls[CU_NUM] __attribute__((depth(NUM_JA_PER_SET)));


__kernel void cosine_int_32(__global const uint* restrict jnt_angles) {

	__local char grad_table_32[4096];
	__local long intercept_table_32[4096];

	// init the LUT of integer-encoded cosine function
	#pragma unroll 2
	for (uint i = 0; i < 351; ++i) {
		grad_table_32[i] = -1;
		intercept_table_32[i] = 287708254;
	}

	#pragma unroll 2
	for (uint i = 351; i < 819; ++i) {
		grad_table_32[i] = 1;
		intercept_table_32[i] = -368140054;
	}

	#pragma unroll 2
	for (uint i = 819; i < 1872; ++i) {
		grad_table_32[i] = 3;
		intercept_table_32[i] = -2262381335;
	}

	#pragma unroll 4
	for (uint i = 1872; i < 2457; ++i) {
		grad_table_32[i] = 1;
		intercept_table_32[i] = 1717986918;
	}

	#pragma unroll 2
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

	// NOTE: INTEL OPENCL does not support dynamic indexing on channel IDs
	#pragma unroll 4
	for (uint idx = 0; idx < (uint)NUM_JA_PER_SET; ++idx) {
		
		uint angle_inputs[16];
		char grads[16];
		long intercepts[16];

		#pragma unroll
		for (int i = 0; i < 16; ++i) {
			angle_inputs[i] = jnt_angles[idx + i * 6];

			uint angle_idx = angle_inputs[i] >> 20;
			grads[i] = grad_table_32[angle_idx];
			intercepts[i] = intercept_table_32[angle_idx];
		}

		ulong trig_vals[16];

		#pragma unroll
		for (int i = 0; i < 16; ++i) {
			trig_vals[i] = (ulong) grads[i] * angle_inputs[i] + intercepts[i];
		}

		// write_channel_intel(trig_val_chan, trig_val_temp_0);
		write_channel_intel(all_trig_val_chnls[0], trig_vals[0]);
		write_channel_intel(all_trig_val_chnls[1], trig_vals[1]);
		write_channel_intel(all_trig_val_chnls[2], trig_vals[2]);
		write_channel_intel(all_trig_val_chnls[3], trig_vals[3]);
		write_channel_intel(all_trig_val_chnls[4], trig_vals[4]);
		write_channel_intel(all_trig_val_chnls[5], trig_vals[5]);
		write_channel_intel(all_trig_val_chnls[6], trig_vals[6]);
		write_channel_intel(all_trig_val_chnls[7], trig_vals[7]);
		write_channel_intel(all_trig_val_chnls[8], trig_vals[8]);
		write_channel_intel(all_trig_val_chnls[9], trig_vals[9]);
		write_channel_intel(all_trig_val_chnls[10], trig_vals[10]);
		write_channel_intel(all_trig_val_chnls[11], trig_vals[11]);
		write_channel_intel(all_trig_val_chnls[12], trig_vals[12]);
		write_channel_intel(all_trig_val_chnls[13], trig_vals[13]);
		write_channel_intel(all_trig_val_chnls[14], trig_vals[14]);
		write_channel_intel(all_trig_val_chnls[15], trig_vals[15]);
	}
}


// __attribute__((num_compute_units(CU_NUM)))
// aggregate all the trig. values from the block-read channel before moving on
__kernel void get_pose_by_jnts_int_32(__global ulong* restrict ee_pose) {
	ulong trig_vals_channeled[NUM_JA_PER_SET];
	int cu_idx = get_global_id(0);
	
	switch (cu_idx) {
		case 0:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[0]);
			}
			break;

		case 1:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[1]);
			}
			break;

		case 2: 
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[2]);
			}
			break;

		case 3:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[3]);
			}
			break;
		
		case 4:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[4]);
			}
			break;
		
		case 5:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[5]);
			}
			break;

		case 6:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[6]);
			}
			break;

		case 7:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[7]);
			}
			break;

		case 8:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[8]);
			}
			break;

		case 9:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[9]);
			}
			break;

		case 10:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[10]);
			}
			break;

		case 11:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[11]);
			}
			break;

		case 12:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[12]);
			}
			break;

		case 13:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[13]);
			}
			break;

		case 14:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[14]);
			}
			break;

		case 15:
			for (int i = 0; i < (int)NUM_JA_PER_SET; ++i) {
				trig_vals_channeled[i] = read_channel_intel(all_trig_val_chnls[15]);
			}
			break;
	}

	int offset = cu_idx * NUM_OUT_POSE_PER_SET;

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


// __attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void get_pose_by_jnts(__global const double* restrict radians,
								__global double* restrict ee_pose) {

	// int cu_idx = get_global_id(0);
	for (int cu_idx = 0; cu_idx < FP_SIMUL_SET; ++cu_idx) {
		int radians_offset = cu_idx * NUM_RAD_PER_SET;
		int out_ee_pose_offset = cu_idx * NUM_OUT_POSE_PER_SET_FP;

		double link_lengths[3] = {sqrt(3.5*3.5+3.9*3.9), sqrt(1.7*1.7+10.5*10.5), sqrt(3.5*3.5+16.5*16.5)};

		// y = base_height/2.9;
		// ee_pose[1] = 2.9;
		double y = 2.9;

		// y += l1*sin(a2) + l2*sin(a3) + l3*sin(a4);

		for (int i = 0; i < 3; ++i) {
			y += link_lengths[i] * cos(radians[radians_offset + i + 5]);
		}


		// d1 = -l2*cos(a3);
		double d1 = -link_lengths[1] * cos(radians[radians_offset + 2]);


		// d1 += l1*cos(a2)+l3*cos(a4);
		for (int i = 0; i < 3; i += 2) {
			d1 += link_lengths[i] * cos(radians[radians_offset + i + 1]);
		}

		ee_pose[out_ee_pose_offset] = d1 * cos(radians[radians_offset + 4]);
		ee_pose[out_ee_pose_offset + 1] = y;
		ee_pose[out_ee_pose_offset + 2] = d1 * cos(radians[radians_offset]);
	}
}
