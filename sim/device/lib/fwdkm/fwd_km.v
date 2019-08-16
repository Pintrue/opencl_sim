`include "trig.v"


`define INIT_JNT_ANGLE_0 0
`define INIT_JNT_ANGLE_1 112855247
`define INIT_JNT_ANGLE_2 7877904265

`define _INIT_JNT_ANGLE_0 -1104420162
`define _INIT_JNT_ANGLE_1 -991564915
`define _INIT_JNT_ANGLE_2 6773484103

`define BASE_HEIGHT_INT 290
`define LINK_LENGTH_INT_1 524
`define LINK_LENGTH_INT_2 1064
`define LINK_LENGTH_INT_3 1687

`define BASE_HEIGHT 2.9
`define LINK_LENGTH_1 5.240229002629561
`define LINK_LENGTH_2 10.636728820459794
`define LINK_LENGTH_3 16.867127793433

`define LINK_LENGTH_SCALE 100.0

`define INT_TRIG_SCALE_RANGE 4294967295


module fwd_km (
    input clock,
    input resetn,
	input unsigned [31:0] jnt_int_0;
	input unsigned [31:0] jnt_int_1;
	input unsigned [31:0] jnt_int_2;
	output unsigned [63:0] ee_pos_x;
	output unsigned [63:0] ee_pos_y;
	output unsigned [63:0] ee_pos_z;
);

    // member variables
	trig TIRG();
    
	reg unsigned [31:0] a1;
	reg unsigned [31:0] a3;
	reg unsigned [31:0] a4;
	reg unsigned [31:0] _a1;
	reg unsigned [31:0] _a3;
	reg unsigned [31:0] _a4;

	reg unsigned [63:0] d1;


	always @ (posedge reset) begin: init_km
		// reset the cache to initial value
		a1 = `INIT_JNT_ANGLE_0;
		a3 = `INIT_JNT_ANGLE_1;
		a4 = `INIT_JNT_ANGLE_2;
		_a1 = `_INIT_JNT_ANGLE_0;
		_a3 = `_INIT_JNT_ANGLE_1;
		_a4 = `_INIT_JNT_ANGLE_2;

		getEEPoseByJntsInt32(jnt_int_0, jnt_int_1, jnt_int_2, ee_pos_x, ee_pos_y, ee_pos_z);
	end


	task getEEPoseByJntsInt32;
		//input and output
		input unsigned [31:0] jnt_int_0;
		input unsigned [31:0] jnt_int_1;
		input unsigned [31:0] jnt_int_2;

		output unsigned [63:0] ee_pos_x;
		output unsigned [63:0] ee_pos_y;
		output unsigned [63:0] ee_pos_z;

		// local variables
		reg unsigned [63:0] d2;
		reg unsigned [63:0] d3;
		reg unsigned [63:0] d4;
		reg unsigned [63:0] d5;
		reg unsigned [63:0] d6;
		reg unsigned [63:0] d7;
		reg unsigned [63:0] d8;
		
		reg [63:0] cos_temp_out;


		begin
			reset = 1'b1;

			#1 a1 = a1 + jnt_int_0;
			a3 = a3 + jnt_int_1;
			a4 = a4 - jnt_int_2 - jnt_int_1;
			_a1 = _a1 + jnt_int_0;
			_a3 = _a3 + jnt_int_1;
			_a4 = _a4 - jnt_int_2 - jnt_int_1;

			d2 = `BASE_HEIGHT_INT;
			d3 = `LINK_LENGTH_INT_1 * 3581808896;	// cosineInt32(_a2) 

			TRIG.cosineInt32(_a3, cos_temp_out);
			d4 = `LINK_LENGTH_INT_2 * cos_temp_out;

			TRIG.cosineInt32(_a4, cos_temp_out);
			d5 = `LINK_LENGTH_INT_3 * cos_temp_out;

			ee_pos_y = d2 + d3 + d4 + d5;

			TRIG.cosineInt32(a4, cos_temp_out);
			d6 = `LINK_LENGTH_INT_3 * cos_temp_out;

			TRIG.cosineInt32(a3, cos_temp_out);
			d7 = `LINK_LENGTH_INT_2 * cos_temp_out;

			d8 = `LINK_LENGTH_INT_1 * 3745731782;	//cosineInt32(a2)

			d1 = d6 - d7 + d8;

			TRIG.cosineInt32(a1, cos_temp_out);
			ee_pos_z = d1 * cos_temp_out;

			TRIG.cosineInt32(_a1, cos_temp_out);
			ee_pos_x = d1 * cos_temp_out;

			// $display("d1 = %d", d1);
			// $display("d2 = %d", d2);
			// $display("d3 = %d", d3);			
			// $display("d4 = %d", d4);
			// $display("d5 = %d", d5);
			// $display("y = %d", ee_pos_y);
			// $display("z = %d", ee_pos_z);
			// $display("x = %d", ee_pos_x);
		end
	endtask

endmodule