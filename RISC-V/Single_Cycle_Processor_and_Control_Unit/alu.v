module alu(a, b, alucontrol, result, zero);

	input signed [63:0] a, b;
	input  [2:0]  alucontrol;
	output reg [63:0] result;
	output zero;

	always @(*) begin
		case (alucontrol)
	      		3'b000:  result = a+b;         // add
	      		3'b001:  result = a-b;         // subtract
	      		3'b010:  result = a & b;       // and
	      		3'b011:  result = a | b;       // or
	      		3'b100:  result = a ^ b;       // xor
	      		3'b101:  result = {63'd0,(a < b)}; // slt
	      		3'b110:  result = a << b[4:0]; // sll
	      		3'b111:  result = a >> b[4:0]; // srl
	      		default: result = 32'bx;
		endcase
	end
		assign zero = (result==0)? 1 : 0;
endmodule
