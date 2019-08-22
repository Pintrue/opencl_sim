#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include "c_api.hpp"
#include "host.hpp"
#include "sim.hpp"
#include "CL/opencl.h"


using namespace std;


static Sim sim;
int _task_flag;
double link_length[3] = {sqrt(3.5*3.5+3.9*3.9), sqrt(1.7*1.7+10.5*10.5), sqrt(3.5*3.5+16.5*16.5)};


matrix_t* resetStateReaching(int rand_angle, int dest_pos, int state_dim, int act_dim);
matrix_t* resetStatePnP(int rand_angle, int dest_pos, int state_dim, int act_dim);
matrix_t* stepReaching(matrix_t* action, int state_dim, int act_dim);
matrix_t* stepPnP(matrix_t* action, int state_dim, int act_dim);
void setReachingRewardBit(double full_state[FULL_STATE_NUM_COLS]);
void setPnPRewardBit(double full_state[FULL_STATE_NUM_COLS]);
bool regulatedJntAngles(const double curr_jnt_angles[NUM_OF_JOINTS],
						const double delta[ACTION_DIM],
						double new_state[FULL_STATE_NUM_COLS]);
bool withinCylinder(double center[3], int radius, double obj[3]);
matrix_t* normalizeAction(matrix_t* action);
matrix_t* reachingRandomAction(int state_dim, int act_dim);
matrix_t* pnpRandomAction(int state_dim, int act_dim);
void revertPose(const ulong ee_pose_int[6], double ee_pose[3]);


matrix_t* new_matrix(int rows, int cols) {
	matrix_t* new_mat = (matrix_t*) malloc(sizeof(matrix_t));

	new_mat->data = (double*) calloc(rows*cols, sizeof(double));	

	new_mat->rows = rows;
	new_mat->cols = cols;
	new_mat->max_size = rows * cols;

	return new_mat;
}


void free_matrix(matrix_t* mat) {
	free(mat->data);
	free(mat);
}


double randUniform(double lower, double upper) {
	assert(upper >= lower);
	return lower + (rand() / (double(RAND_MAX) / (upper - lower)));
}


int initEnv(int act_dim, int task_flag) {
	if (task_flag == REACHING_TASK_FLAG) {
		sim = Sim();
	} else if (task_flag == PICK_N_PLACE_TASK_FLAG) {
		sim = Sim();
	} else {
		cout << "ERROR: Unrecognized flag encountered at " << __LINE__
			<< " in file " << __FILE__; 
		exit(-1);
	}

	_task_flag = task_flag;

	cl_int err;
	if (!initOpencl()) {
		printf("ERROR: Unable to initialize OpenCL at %s: line %d.\n", __FILE__, __LINE__);
		return -1;
	}

	return 0;
}


matrix_t* resetState(int rand_angle, int dest_pos, int state_dim, int act_dim) {
	matrix_t* ret;

	if (_task_flag == REACHING_TASK_FLAG) {
		ret = resetStateReaching(rand_angle, dest_pos, state_dim, act_dim);
	} else if (_task_flag == PICK_N_PLACE_TASK_FLAG) {
		ret = resetStatePnP(rand_angle, dest_pos, state_dim, act_dim);
	} else {
		cout << "ERROR: Unrecognized flag encountered at " << __LINE__
			<< " in file " << __FILE__; 
		exit(-1);
	}

	return ret;
}


matrix_t* resetStateReaching(int rand_angle, int dest_pos, int state_dim, int act_dim) {
	printf("in reaching reset\n");
	matrix_t* ret = new_matrix(1, FULL_STATE_NUM_COLS);
	double* data = ret->data;

	sim._num_of_steps = 0;


	// set initial angle configurations
	if (rand_angle == 1) {
		data[0] = randUniform(JNT0_L, JNT0_U);
		data[1] = randUniform(JNT1_L, JNT1_U);
		data[2] = randUniform(JNT2_L, JNT2_U);
	} else {
		// for (int i = 0; i < NUM_OF_JOINTS; ++i) {
		// 	data[i] = 0;
		// }
		
		data[0] = -0.502065;
		data[1] = -0.675970;
		data[2] = -1.911503;
	}


	double ee_pos[3];

	for (int i = 0; i < NUM_OF_JOINTS; ++i) {
		sim._init_joint_angles[i] = data[i];
		sim._curr_joint_angles[i] = data[i];
	}


	// TODO: get pos from kernel
	cout << "Before initializing the input to kernel" << endl;

	initInput();
	cout << "Finish trig. init" << endl;
	initKMInput();

	cout << "Before entering kernel execution" << endl;

	run();
	runKM();

	cout << "Result from KM kernel execution" << endl;
	for (int i = 0; i < 6; ++i) {
		cout << output_ee_pose[i] << " ";
	}
	cout << endl;

	cout << "Finish kernel execution" << endl;

	revertPose(output_ee_pose, ee_pos);

	// set initial arm pose
	for (int i = 0; i < 3; ++i) {
		data[REACHING_FST_EE_POS_OFFSET + i] = ee_pos[i];
		data[REACHING_SND_EE_POS_OFFSET + i] = ee_pos[i];
	}


	// set destination position
	if (dest_pos == 1) {
		data[REACHING_DEST_POS_OFFSET + 0] = randUniform(REACHING_WORKSPACE_X_LOWER, REACHING_WORKSPACE_X_UPPER);
		data[REACHING_DEST_POS_OFFSET + 1] = 0;
		data[REACHING_DEST_POS_OFFSET + 2] = randUniform(REACHING_WORKSPACE_Z_LOWER, REACHING_WORKSPACE_Z_UPPER);
	} else {
		data[REACHING_DEST_POS_OFFSET] = 0;
		data[REACHING_DEST_POS_OFFSET + 1] = 0;
		data[REACHING_DEST_POS_OFFSET + 2] = 17.33;
	}


	// set termination and reward bits
	data[REACHING_TERMINAL_BIT_OFFSET] = 0;	// terminal flag
	data[REACHING_REWARD_BIT_OFFSET] = -1;	// reward bit

	return ret;
}


matrix_t* resetStatePnP(int rand_angle, int dest_pos, int state_dim, int act_dim) {
	matrix_t* ret = new_matrix(1, FULL_STATE_NUM_COLS);
	double* data = ret->data;

	sim._num_of_steps = 0;

	// set initial angle configurations
	if (rand_angle == 1) {
		data[0] = randUniform(JNT0_L, JNT0_U);
		data[1] = randUniform(JNT1_L, JNT1_U);
		data[2] = randUniform(JNT2_L, JNT2_U);
	} else {
		for (int i = 0; i < NUM_OF_JOINTS; ++i) {
			data[i] = 0;
		}
	}

	double ee_pos[3];

	for (int i = 0; i < 3; ++i) {
		sim._init_joint_angles[i] = data[i];
		sim._curr_joint_angles[i] = data[i];
	}


	// TODO: get pos from kernel

	// set initial arm pose
	for (int i = 0; i < 3; ++i) {
		data[PNP_EE_POS_OFFSET + i] = ee_pos[i];
	}

	
	data[PNP_EE_STATE_OFFSET] = 0;


	// set object position
	data[PNP_FST_OBJ_POS_OFFSET] = randUniform(PNP_OBJ_X_LOWER, PNP_OBJ_X_UPPER);
	data[PNP_FST_OBJ_POS_OFFSET + 1] = PNP_OBJ_HEIGHT;
	data[PNP_FST_OBJ_POS_OFFSET + 2] = randUniform(PNP_OBJ_Z_LOWER, PNP_OBJ_Z_UPPER);

	for (int i = 0; i < 3; ++i) {
		data[PNP_SND_OBJ_POS_OFFSET + i] = data[PNP_FST_OBJ_POS_OFFSET + i];
		sim._init_obj[i] = data[PNP_FST_OBJ_POS_OFFSET + i];
		sim._obj[i] = data[PNP_FST_OBJ_POS_OFFSET + i];
	}

	
	// set object attachment status
	data[PNP_HAS_OBJ_OFFSET] = 0;


	// set destination position
	if (dest_pos == 1) {
		data[PNP_DEST_POS_OFFSET] = randUniform(PNP_WORKSPACE_X_LOWER, PNP_WORKSPACE_X_UPPER);
		data[PNP_DEST_POS_OFFSET + 1] = 0;
		data[PNP_DEST_POS_OFFSET + 2] = randUniform(PNP_WORKSPACE_Z_LOWER, PNP_WORKSPACE_Z_UPPER);
	} else {
		data[PNP_DEST_POS_OFFSET] = -7.495477e+00;
		data[PNP_DEST_POS_OFFSET + 1] = 0.000000e+00;
		data[PNP_DEST_POS_OFFSET + 2] = 1.870172e+01;
	}

	for (int i = 0; i < 3; ++i) {
		sim._dest[i] = data[PNP_DEST_POS_OFFSET + i];
	}


	// set termination and reward bits
	data[PNP_TERMINAL_BIT_OFFSET] = 0;	// terminal flag
	data[PNP_REWARD_BIT_OFFSET] = -1;	// reward bit
}


matrix_t* denormalizeAction(matrix_t* action) {
	matrix_t* ret = new_matrix(action->rows, action->cols);

	for (int i = 0; i < 3; ++i) {
		ret->data[i] = (double) (action->data[i] + 1) / (double) 2 * (ACTION_BOUND_UPPER - ACTION_BOUND_LOWER) + ACTION_BOUND_LOWER;
	}

	return ret;
}


bool regulatedJntAngles(const double curr_jnt_angles[NUM_OF_JOINTS],
						const double delta[ACTION_DIM],
						double new_state[FULL_STATE_NUM_COLS]) {

	double res_jnt_angles[NUM_OF_JOINTS];

	for (int i = 0; i < NUM_OF_JOINTS; ++i) {
		res_jnt_angles[i] = curr_jnt_angles[i] + delta[i];
	}

	new_state[0] = min(max(res_jnt_angles[0], JNT0_L), JNT0_U);
	new_state[1] = min(max(res_jnt_angles[1], JNT1_L), JNT1_U);
	new_state[2] = min(max(res_jnt_angles[2], JNT2_L), JNT2_U);

	return (res_jnt_angles[0] == new_state[0])
			&& (res_jnt_angles[1] == new_state[1])
			&& (res_jnt_angles[2] == new_state[2]);
}


void setReachingRewardBit(double full_state[FULL_STATE_NUM_COLS]) {
	double diff = 0;

	for (int i = 0; i < 3; ++i) {
		double delta = full_state[REACHING_DEST_POS_OFFSET + i]
						- full_state[REACHING_FST_EE_POS_OFFSET + i];
		diff += delta * delta;
	}

	bool at_dest = sqrt(diff) <= REACHING_EE_AT_DEST_RANGE;
	full_state[REACHING_REWARD_BIT_OFFSET] = at_dest ? 0 : -1;
}


void setPnPRewardBit(double full_state[FULL_STATE_NUM_COLS]) {
	double diff = 0;

	for (int i = 0; i < 3; ++i) {
		double delta = full_state[PNP_FST_OBJ_POS_OFFSET + i]
						- full_state[PNP_DEST_POS_OFFSET + i];
		diff += delta * delta;
	}

	bool obj_at_dest = sqrt(diff) <= PNP_OBJ_AT_DEST_RANGE;
	full_state[PNP_REWARD_BIT_OFFSET] = (obj_at_dest && full_state[PNP_EE_STATE_OFFSET])
											? 0 : -1;
}


matrix_t* step(matrix_t* action, int state_dim, int act_dim) {
	matrix_t* ret;

	if (_task_flag == REACHING_TASK_FLAG) {
		ret = stepReaching(action, state_dim, act_dim);
	} else if (_task_flag == PICK_N_PLACE_TASK_FLAG) {
		ret = stepPnP(action, state_dim, act_dim);
	} else {
		cout << "ERROR: Unrecognized flag encountered at " << __LINE__
			<< " in file " << __FILE__; 
		exit(-1);
	}

	return ret;
}


matrix_t* stepReaching(matrix_t* action, int state_dim, int act_dim) {
	matrix_t* ret = new_matrix(1, FULL_STATE_NUM_COLS);
	double* data = ret->data;

	matrix_t* denormed_mat = denormalizeAction(action);
	regulatedJntAngles(sim._curr_joint_angles, denormed_mat->data, data);

	// set the new angle after action is applied
	for (int i = 0; i < NUM_OF_JOINTS; ++i) {
		sim._curr_joint_angles[i] = data[i];
	}

	sim._num_of_steps += 1;
	double ee_pos[3];


	// TODO: get pos from kernel


	for (int i = 0; i < 3; ++i) {
		data[REACHING_FST_EE_POS_OFFSET + i] = ee_pos[i];
		data[REACHING_SND_EE_POS_OFFSET + i] = ee_pos[i];
	}

	for (int i = 0; i < 3; ++i) {
		data[REACHING_DEST_POS_OFFSET + i] = sim._dest[i];
	}

	if (sim._num_of_steps >= 50) {
		data[REACHING_TERMINAL_BIT_OFFSET] = 1;
	}

	setReachingRewardBit(data);
	free_matrix(denormed_mat);

	return ret;
}


bool ifObjBecomesAttached(double full_state[FULL_STATE_NUM_COLS]) {
	double diff = 0;

	for (int i = 0; i < 3; ++i) {
		double delta = full_state[PNP_EE_POS_OFFSET + i]
						- full_state[PNP_FST_OBJ_POS_OFFSET + i];
		diff += delta * delta;
	}

	return sqrt(diff) <= PNP_OBJ_ATTACHED_RANGE;
}


bool withinCylinder(double center[3], int radius, double obj[3]) {
	double distToCenter = sqrt(pow(center[0] - obj[2], 2)
							+ pow(center[2] - obj[2], 2));
	return distToCenter <= (double) radius;
}


matrix_t* stepPnP(matrix_t* action, int state_dim, int act_dim) {
	matrix_t* ret = new_matrix(1, FULL_STATE_NUM_COLS);
	double* data = ret->data;

	matrix_t* denormed_mat = denormalizeAction(action);
	denormed_mat->data[3] = action->data[3];

	regulatedJntAngles(sim._curr_joint_angles, denormed_mat->data, data);

	// set the new angle after action is applied
	for (int i = 0; i < NUM_OF_JOINTS; ++i) {
		sim._curr_joint_angles[i] = data[i];
	}

	sim._num_of_steps += 1;
	double ee_pos[3];


	// TODO: get pos from kernel

	for (int i = 0; i < 3; ++i) {
		data[PNP_EE_POS_OFFSET] = ee_pos[i];
	}

	// set the state of end-effector
	data[PNP_EE_STATE_OFFSET] = round(action->data[3]);

	// set the attachment status of the object
	if (sim._has_obj && data[PNP_EE_STATE_OFFSET == 1]) {
		for (int i = 0; i < 3; ++i) {
			sim._obj[i] = data[PNP_EE_POS_OFFSET + i];
		}

		data[PNP_HAS_OBJ_OFFSET] = 1;
		sim._ee_state = true;
	} else if (sim._has_obj && data[PNP_EE_STATE_OFFSET] == 0) {
		sim._obj[1] = PNP_OBJ_HEIGHT;
		sim._has_obj = false;
		data[PNP_HAS_OBJ_OFFSET] = 0;
	} else if ((!sim._has_obj) && data[PNP_EE_STATE_OFFSET] == 1) {
		for (int i = 0; i < 3; ++i) {
			data[PNP_FST_OBJ_POS_OFFSET + i] = sim._obj[i];
		}

		if (ifObjBecomesAttached(data)) {
			data[PNP_HAS_OBJ_OFFSET] = 1;
			
			for (int i = 0; i < 3; ++i) {
				sim._obj[i] = data[PNP_EE_POS_OFFSET + i];
			}

			sim._has_obj = true;
			data[PNP_HAS_OBJ_OFFSET] = 1;
		} else {
			// do nothing
		}
		sim._ee_state = true;
	}

	if (!withinCylinder(sim._init_obj, PNP_OBJ_LIFT_UP_CYLINDER_RADIUS, ee_pos)
		&& (!withinCylinder(sim._dest, PNP_OBJ_LIFT_UP_CYLINDER_RADIUS, ee_pos))
		&& ee_pos[1] < PNP_OBJ_AFLOAT_MIN_HEIGHT) {
		
		sim._obj[1] = PNP_OBJ_HEIGHT;
		data[PNP_HAS_OBJ_OFFSET] = 0;
		sim._has_obj = false;
	}

	// set obj pos
	for (int i = 0; i < 3; ++i) {
		data[PNP_FST_OBJ_POS_OFFSET + i] = sim._obj[i];
		data[PNP_SND_OBJ_POS_OFFSET + i] = sim._obj[i];
	}

	for (int i = 0; i < 3; ++i) {
		data[PNP_DEST_POS_OFFSET + i] = sim._dest[i];
	}

	if (sim._num_of_steps >= 50) {
		data[PNP_TERMINAL_BIT_OFFSET] = 1;
	}

	setPnPRewardBit(data);
	free_matrix(denormed_mat);

	return ret;
}


void closeEnv(int state_dim, int act_dim) {
	cleanup();

	return;	
}


matrix_t* normalizeAction(matrix_t* action) {
	matrix_t* ret = new_matrix(action->rows, action->cols);

	ret->data[0] = (double) (action->data[0] - JNT0_L) / (JNT0_U - JNT0_L) * 2 - 1;
	ret->data[1] = (double) (action->data[1] - JNT1_L) / (JNT1_U - JNT1_L) * 2 - 1;
	ret->data[2] = (double) (action->data[2] - JNT2_L) / (JNT2_U - JNT2_L) * 2 - 1;

	return ret;
}


matrix_t* reachingRandomAction(int state_dim, int act_dim) {
	matrix_t* ret = new_matrix(1, act_dim);
	double* data = ret->data;

	double to_ja[3];
	to_ja[0] = randUniform(JNT0_L, JNT0_U);
	to_ja[1] = randUniform(JNT1_L, JNT1_U);
	to_ja[2] = randUniform(JNT2_L, JNT2_U);

	for (int i = 0; i < NUM_OF_JOINTS; ++i) {
		data[i] = to_ja[i] - sim._curr_joint_angles[i];
	}

	matrix_t* norm_ret = normalizeAction(ret);
	free_matrix(ret);

	return norm_ret;
}


matrix_t* pnpRandomAction(int state_dim, int act_dim) {
	matrix_t* ret = new_matrix(1, act_dim);
	double* data = ret->data;

	matrix_t* ja_action = reachingRandomAction(0, act_dim - 1);
	for (int i = 0; i < act_dim - 1; ++i) {
		data[i] = ja_action->data[i];
	}

	data[3] = (sim._ee_state) ? 1 : 0;
	free_matrix(ja_action);

	return ret;
}


void revertPose(const ulong ee_pose_int[6], double ee_pose[3]) {
	double m = INT_TRIG_SCALE_RANGE / 2.0;
	double b = INT_TRIG_SCALE_RANGE / 2.0;

	double d1 = (ee_pose_int[3] / 100.0
				- (link_length[3] - link_length[2] + link_length[1]) * b) / m;

	ee_pose[2] = d1 * convertTrigEncToVal(ee_pose_int[4]);
	ee_pose[0] = d1 * convertTrigEncToVal(ee_pose_int[5]);
	
	ee_pose[1] = ee_pose_int[1] / 100.0 - 2.9;
	ee_pose[1] -= (link_length[1] + link_length[2] + link_length[3]) * b;
	ee_pose[1] = ee_pose[1] / m + 2.9;
}


int main() {
	initEnv(ACTION_DIM, REACHING_TASK_FLAG);
	matrix_t* fs = resetState(0, 0, 0, 0);

	double* data = fs->data;
	for (int i = 0; i < fs->rows; ++i) {
		for (int j = 0; j < fs->cols; ++j) {
			cout << *(data + i * fs->cols + j) << " ";
		}
	}
	cout << endl;

	closeEnv(0, 0);
}
