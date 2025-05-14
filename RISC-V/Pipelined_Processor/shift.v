module shift (
input [63:0] In,
output [63:0] out);

assign out = In << 1;
endmodule 
