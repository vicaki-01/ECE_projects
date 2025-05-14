module WBU (input [63:0] MemData,ALUResult, input Mem2Reg, 
output [63:0] Writeback 
);


MUX M1(ALUResult,MemData, Mem2Reg,Writeback);

endmodule
