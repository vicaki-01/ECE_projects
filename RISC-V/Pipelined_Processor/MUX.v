module MUX (input [63 : 0] a,b,
 input  s,
 output [63 : 0] out);

assign out = (s == 1'b0) ? a : b;
endmodule

