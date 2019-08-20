#include <stdlib.h>
#include <stdio.h>
#include "c_api.hpp"
#include "sim.hpp"


using namespace std;


static Sim sim;
int _task_flag;

matrix_t* resetStateReaching(int rand_angle, int dest_pos, int state_dim, int act_dim);
matrix_t* resetStatePickNPlace(int rand_angle, int dest_pos, int state_dim, int act_dim);

matrix_t* new_matrix(int rows, int cols) {
	matrix_t* new_mat = (matrix_t*) malloc(sizeof(matrix_t));
	
	new_mat->rows = rows;
	new_mat->cols = cols;
	new_mat->max_size = rows * cols;

	return new_mat;
}


double randAngleRads(double lower, double upper) {
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
	return 0;
}


matrix_t* resetState(int rand_angle, int dest_pos, int state_dim, int act_dim) {
	matrix_t* ret;

	if (_task_flag == REACHING_TASK_FLAG) {
		ret = resetStateReaching(rand_angle, dest_pos, state_dim, act_dim);
	} else if (_task_flag == PICK_N_PLACE_TASK_FLAG) {
		ret = resetStatePickNPlace(rand_angle, dest_pos, state_dim, act_dim);
	} else {
		cout << "ERROR: Unrecognized flag encountered at " << __LINE__
			<< " in file " << __FILE__; 
		exit(-1);
	}

	return ret;
}


matrix_t* resetStateReaching(int rand_angle, int dest_pos, int state_dim, int act_dim) {
	matrix_t* ret = new_matrix(1, FULL_STATE_NUM_COLS);
	double* data = ret->data;

	sim._num_of_steps = 0;

	if (rand_angle == 1) {
		data[0] = randAngleRads(JNT0_L, JNT0_U);
		data[1] = randAngleRads(JNT1_L, JNT1_U);
		data[2] = randAngleRads(JNT2_L, JNT2_U);
	} else {
		for (int i = 0; i < NUM_OF_JOINTS; ++i) {
			data[i] = 0;
		}
	}

	double ee_pos[6];

	for (int i = 0; i < NUM_OF_JOINTS; ++i) {
		sim._init_joint_angles[i] = data[i];
		sim._joint_angles[i] = data[i];
	}


	// TODO: get angle from kernel


	for (int i = 0; i < 3; ++i) {
		data[i + REACHING_FST_EE_POS_OFFSET] = ee_pos[i];
		data[i + REACHING_SND_EE_POS_OFFSET] = ee_pos[i];
	}

	if (dest_pos == 1) {
		matrix_t* dest = new_matrix(1, 3);
		double* dest_data = dest->data;
	}
}

matrix_t* resetStatePickNPlace(int rand_angle, int dest_pos, int state_dim, int act_dim);


matrix_t* denormalizeAction(matrix_t* action);

matrix_t* step(matrix_t* action, int state_dim, int act_dim);

void closeEnv(int state_dim, int act_dim);

matrix_t* randomAction(int state_dim, int act_dim);