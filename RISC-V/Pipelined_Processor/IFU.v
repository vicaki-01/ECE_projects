module IFU (input clk,reset,
input [63:0] initialPC,
input AddSel,PC_En,input [63 : 0] PC_Plus4, input [63 : 0] BranchTargetAdd, 
output [63 : 0] PC, 
output [31:0] Instruction);

wire [63:0] PC1;

MUX  PC_Mux(PC_Plus4,BranchTargetAdd,AddSel,PC1);
PCCounter PC_Register(clk,reset, initialPC, PC_En,PC1,PC);
adder  PC_Adder(PC,PC_Plus4);
instructionmemory IM(PC,Instruction);

//end 
endmodule 
