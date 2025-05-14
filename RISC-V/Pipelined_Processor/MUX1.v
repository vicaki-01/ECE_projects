module MUX1 (input [63 : 0] a,b,c,
 input  [1:0] s,
 output [63 : 0] out);
assign out = (s == 2'b10) ? c : (s == 2'b01) ? b : a;
endmodule
