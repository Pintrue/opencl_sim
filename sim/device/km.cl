__kernel void cosine_int_32(__global const uint* restrict jnt_angles,
							__global long* restrict output) {

	local char grad_table_32[4095];
	local int intercept_table_32[4095];

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
	uint angle_idx = (angle_input & 0xFFF0_0000) >> 20;

	// obtain trigonometry encoding value at that LUT index
	output[idx] = grad_table_32[angle_idx] * angle_input + intercept_table_32[angle_idx];
}