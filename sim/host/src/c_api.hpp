#ifndef C_API_HPP
#define C_API_HPP


typedef struct _matrix_t {
    double* data;
    int rows, cols;
    int max_size;
} matrix_t;


#ifndef M_PI
    #define M_PI 3.14159265359
#endif

#define JNT0_L -M_PI/2
#define JNT0_U M_PI/2
#define JNT1_L 0.0
#define JNT1_U 130.0/180.0*M_PI
#define JNT2_L -M_PI/2
#define JNT2_U 0.0


#define REACHING
// #define PICK_N_PLACE

#ifdef REACHING
    #define FULL_STATE_NUM_COLS 14
    #define ACTION_DIM 3
#endif

#ifndef REACHING
    #ifdef PICK_N_PLACE
        #define FULL_STATE_NUM_COLS 19
        #define ACTION_DIM 4
    #endif
#endif

#define REACHING_TASK_FLAG 100
#define PICK_N_PLACE_TASK_FLAG 200

#define REACHING_FST_EE_POS_OFFSET 3
#define REACHING_DEST_POS_OFFSET (REACHING_FST_EE_POS_OFFSET + 3)
#define REACHING_SND_EE_POS_OFFSET (REACHING_DEST_POS_OFFSET + 3)
#define REACHING_TERMINAL_BIT_OFFSET (REACHING_SND_EE_POS_OFFSET + 3)
#define REACHING_REWARD_BIT_OFFSET (REACHING_TERMINAL_BIT_OFFSET + 1)

#define REACHING_WORKSPACE_X_LOWER -10.5
#define REACHING_WORKSPACE_X_UPPER 10.5
#define REACHING_WORKSPACE_Z_LOWER 12.5
#define REACHING_WORKSPACE_Z_UPPER 22.5
#define REACHING_EE_AT_DEST_RANGE 1.0

#define PNP_EE_POS_OFFSET 3
#define PNP_EE_STATE_OFFSET (PNP_EE_POS_OFFSET + 3)
#define PNP_FST_OBJ_POS_OFFSET (PNP_EE_STATE_OFFSET + 1)
#define PNP_HAS_OBJ_OFFSET (PNP_FST_OBJ_POS_OFFSET + 3)
#define PNP_DEST_POS_OFFSET (PNP_HAS_OBJ_OFFSET + 1)
#define PNP_SND_OBJ_POS_OFFSET (PNP_DEST_POS_OFFSET + 3)
#define PNP_TERMINAL_BIT_OFFSET (PNP_SND_OBJ_POS_OFFSET + 3)
#define PNP_REWARD_BIT_OFFSET (PNP_TERMINAL_BIT_OFFSET + 1)

#define PNP_OBJ_X_LOWER -10.5
#define PNP_OBJ_X_UPPER 10.5
#define PNP_OBJ_Z_LOWER 12.5
#define PNP_OBJ_Z_UPPER 19.5
#define PNP_OBJ_HEIGHT 0.5

#define PNP_WORKSPACE_X_LOWER -10.5
#define PNP_WORKSPACE_X_UPPER 10.5
#define PNP_WORKSPACE_Z_LOWER 12.5
#define PNP_WORKSPACE_Z_UPPER 19.5
#define PNP_OBJ_AT_DEST_RANGE 3.0
#define PNP_OBJ_ATTACHED_RANGE 1.0
#define PNP_OBJ_LIFT_UP_CYLINDER_RADIUS 3.0 // the cylinder within which the object can be lifted up
#define PNP_OBJ_AFLOAT_MIN_HEIGHT 1.5

#define ACTION_BOUND_LOWER -0.0872664626
#define ACTION_BOUND_UPPER 0.0872664626



int initEnv(int act_dim, int task_flag);

// double rand_uniform(double low, double high);

matrix_t* resetState(int rand_angle, int dest_pos, int state_dim, int act_dim);

matrix_t* denormalizeAction(matrix_t* action);

matrix_t* step(matrix_t* action, int state_dim, int act_dim);

void closeEnv(int state_dim, int act_dim);

matrix_t* randomAction(int state_dim, int act_dim);


#endif