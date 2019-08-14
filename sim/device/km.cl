__kernel void cosine_int_32(__global const uint* restrict jnt_angles,
							__global long* restrict output) {

	__local char grad_table_32[4095];
	__local long intercept_table_32[4095];

	// init the LUT of integer-encoded cosine function
	for (unsigned i = 0; i < 351; ++i) {
		grad_table_32[i] = -1;
		intercept_table_32[i] = 287708254;
	}

	for (unsigned i = 351; i < 819; ++i) {
		grad_table_32[i] = 1;
		intercept_table_32[i] = -368140054;
	}

	for (unsigned i = 819; i < 1872; ++i) {
		grad_table_32[i] = 3;
		intercept_table_32[i] = -2262381335;
	}

	for (unsigned i = 1872; i < 2457; ++i) {
		grad_table_32[i] = 1;
		intercept_table_32[i] = 1717986918;
	}

	for (unsigned i = 2457; i < 3042; ++i) {
		grad_table_32[i] = -1;
		intercept_table_32[i] = 6871947672;
	}

	for (unsigned i = 3042; i < 4096; ++i) {
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
	output[idx] = (long) grad_table_32[angle_idx] * angle_input + intercept_table_32[angle_idx];
}


__kernel void get_pose_by_jnts_int_32(__global const long* restrict trig_vals,
										__global long* ee_pose) {
	__local long link_lengths[4];

	// Link lengths in integer-encoding
	link_lengths[0] = 290;	// base height
	link_lengths[1] = 524;
	link_lengths[2] = 1064;
	link_lengths[3] = 1687;

	long d2 = link_lengths[0];
	long d3 = link_lengths[1] * 3581808896;		// l1 * sin(a2)
	long d4 = link_lengths[2] * trig_vals[4];	// l2 * sin(a3)
	long d5 = link_lengths[3] * trig_vals[5];	// l3 * sin(a4)

	// Aggregate the four sections above to obtain Y-coordinate
	ee_pose[1] = d2 + d3 + d4 + d5;

	long d6 = link_lengths[3] * trig_vals[2];	// l3 * cos(a4)
	long d7 = link_lengths[2] * trig_vals[1];	// l2 * cos(a3)
	long d8 = link_lengths[1] * 3745731782; 	// l1 * cos(a2)

	long d1 = d6 - d7 + d8;

	// Use base angle to obtain X- and Z-coordinates
	ee_pose[0] = d1 * trig_vals[3];
	ee_pose[2] = d1 * trig_vals[0];
}