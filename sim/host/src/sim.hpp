#ifndef SIM_HPP
#define SIM_HPP


#define NUM_OF_JOINTS 3


class Sim {
    public:
        // Constructors
        Sim();
        // Sim(double origin[3]);

        // Simulation status
        int _num_of_steps;      // # of times sim has been forwarded

        // Arm variables
        double _init_joint_angles[NUM_OF_JOINTS];   // initial angle
        double _curr_joint_angles[NUM_OF_JOINTS];        // angles between segments
        double _cart_pose[3];                       // end-effector pos arm is at
        bool _ee_state;                             // whether magnet is charged

        // Environment variables
        bool _has_obj;          // whether object is attached to magnet
        double _dest[3];        // destination pos arm tries to reach
        double _obj[3];         // current obj pos
        double _init_obj[3];    // initial obj pos when spawned
};

#endif