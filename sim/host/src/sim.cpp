#include "sim.hpp"


Sim::Sim() {
    for (int i = 0; i < NUM_OF_JOINTS; ++i) {
        _curr_joint_angles[i] = 0;
    }

    _num_of_steps = 0;
    _has_obj = false;
    _ee_state = false;
}


// Sim::Sim(double origin[3]) {
//     for (int i = 0; i < NUM_OF_JOINTS; ++i) {
//         _curr_joint_angles[i] = 0;
//     }

//     _num_of_steps = 0;
//     _has_obj = false;
//     _ee_state = false;
// }