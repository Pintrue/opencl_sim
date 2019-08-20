#ifndef C_API_HPP
#define C_API_HPP


typedef struct _matrix_t {
    double* data;
    int rows, cols;
    int max_size;
} matrix_t;


#define JNT0_L -M_PI/2
#define JNT0_U M_PI/2
#define JNT1_L 0.0
#define JNT1_U 130.0/180.0*M_PI
#define JNT2_L -M_PI/2
#define JNT2_U 0.0


// #define PICK_N_PLACE
#define REACHING

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

#define PNP_EE_POS_OFFSET 3
#define PNP_EE_STATE_OFFSET (PNP_EE_POS_OFFSET + 3)
#define PNP_FST_OBJ_POS_OFFSET (PNP_EE_STATE_OFFSET + 1)
#define PNP_HAS_OBJ_OFFSET (PNP_FST_OBJ_POS_OFFSET + 3)
#define PNP_DEST_POS_OFFSET (PNP_HAS_OBJ_OFFSET + 1)
#define PNP_SND_OBJ_POS_OFFSET (PNP_DEST_POS_OFFSET + 3)
#define PNP_TERMINAL_BIT_OFFSET (PNP_SND_OBJ_POS_OFFSET + 3)
#define PNP_REWARD_BIT_OFFSET (PNP_TERMINAL_BIT_OFFSET + 1)


int initEnv(int act_dim, int task_flag);

// double rand_uniform(double low, double high);

matrix_t* resetState(int rand_angle, int dest_pos, int state_dim, int act_dim);

matrix_t* denormalizeAction(matrix_t* action);

matrix_t* step(matrix_t* action, int state_dim, int act_dim);

void closeEnv(int state_dim, int act_dim);

matrix_t* randomAction(int state_dim, int act_dim);


#endif