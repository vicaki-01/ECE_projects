module MeMU (input clk, 
input [63:0] ALUResult, RD2, 
input zero, 
input [5:0] controlsignal,
output [63:0] DataMemoryOut,
output Asel 
);


and BranchAnd(Asel,zero,controlsignal[1]);
DataMemory DM(clk,controlsignal[2],controlsignal[3],ALUResult,RD2,DataMemoryOut);

endmodule 
 
