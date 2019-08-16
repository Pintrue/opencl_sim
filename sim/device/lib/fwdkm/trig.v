`ifndef TRIG_V
`define TRIG_V


`define RADIANS_LUT_SELECT_MASK_32 32'hFFF00000
`define RADIANS_LUT_SHIFT_OFFSET_32 20
`define TRIG_ENC_MAX 4294967295
`define TRIG_ENC_MIN 0.0
`define TRIG_ENC_RANGE 4294967295
`define RAD_SCALE_MIN -3.665191429	// -210 degrees
`define RAD_SCALE_MAX 2.443460953	// 140 degrees
`define RAD_SCALE_RANGE 6.108652382 // 350 degrees
`define INT_RAD_SCALE_RANGE 4294967295

`define REGULATE_MIN_MAX(x) ((((x) < 0) ? 0 : (x)) > `TRIG_ENC_MAX ? `TRIG_ENC_MIN : x)


module trig();

	reg signed [7:0] gradTable32[0:4095];
	reg signed [63:0] interceptTable32[0:4095];

	initial begin: init_lut
		integer i;
		for (i = 0; i < 351; i = i + 1) begin
			gradTable32[i] = -1;
			interceptTable32[i] = 287708254;
		end

		for (i = 351; i < 819; i = i + 1) begin
			gradTable32[i] = 1;
			interceptTable32[i] = -368140054;
		end

		for (i = 819; i < 1872; i = i + 1) begin
			gradTable32[i] = 3;
			interceptTable32[i] = -2262381335;
		end

		for (i = 1872; i < 2457; i = i + 1) begin
			gradTable32[i] = 1;
			interceptTable32[i] = 1717986918;
		end

		for (i = 2457; i < 3042; i = i + 1) begin
			gradTable32[i] = -1;
			interceptTable32[i] = 6871947672;
		end

		for (i = 3042; i < 4096; i = i + 1) begin
			gradTable32[i] = -3;
			interceptTable32[i] = 13099500927;
		end
	end // end 'initial' block


	function real toVal32;
	input [63:0] enc;
	begin
		enc = (enc < `TRIG_ENC_MIN || enc > `TRIG_ENC_MAX) ? `REGULATE_MIN_MAX(enc) : enc;
		toVal32 = (enc - `TRIG_ENC_MIN) / (`TRIG_ENC_RANGE) * 2.0 - 1.0;
	end
	endfunction


	task cosineInt32;
		// input and output
		input signed [32:0] radians_int_in; // add one bit in front to prevent overflow
		output signed [63:0] trig_out;

		// local variable
		reg [11:0] lookup_index;

		begin
			lookup_index = (radians_int_in & `RADIANS_LUT_SELECT_MASK_32) >> `RADIANS_LUT_SHIFT_OFFSET_32;
			// $display("before masked: %b", radians_int_in);
			// $display("after masked: %b and in decimal: %d", lookup_index, lookup_index);

			trig_out = gradTable32[lookup_index] * radians_int_in + interceptTable32[lookup_index];
			// $display("m = %d, b = %d and encoding = %d", gradTable32[lookup_index], interceptTable32[lookup_index], radians_int_in);
			// $display("output encoding = %d", trig_out);

		end
	endtask


	task convertRadToInt;
		//input and output
		input real radians_in;
		output unsigned [32:0] radians_int_out;

		begin
			if (radians_in < `RAD_SCALE_MIN || radians_in > `RAD_SCALE_MAX) begin
				$display("Input radians: %f  at 'convertRadToInt' is out of range.", radians_in);
				$finish;
			end else begin
				// $display("radians_in = %f and RAD_SCALE_MIN = %f", radians_in, `RAD_SCALE_MIN);
				// $display("(radians_in - RAD_SCALE_MIN) = %f", radians_in - `RAD_SCALE_MIN);
				radians_int_out = (radians_in - `RAD_SCALE_MIN) / `RAD_SCALE_RANGE * `INT_RAD_SCALE_RANGE;
			end
		end
	endtask
	
endmodule

`endif