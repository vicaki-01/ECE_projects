//-----------------------------------------------------------
//-----------------------------------------------------------
//----------* The Processor with Data and Control*-----------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-------Processor Is Divided to Data Path and Control-------


module riscvsingle(clk, reset,
                   PC,
                   Instr,
                   MemWrite,
                   ALUResult, WriteData,
                   ReadData);

input clk, reset;
output [63:0] PC;
input  [31:0] Instr;
output MemWrite;
output [63:0] ALUResult, WriteData;
input  [63:0] ReadData;

  wire       ALUSrc, RegWrite, Jump, Zero;
  wire [1:0] ResultSrc, ImmSrc;
  wire [2:0] ALUControl;

  controller c(Instr[6:0], Instr[14:12], Instr[30], Zero,
               ResultSrc, MemWrite, PCSrc,
               ALUSrc, RegWrite, Jump,
               ImmSrc, ALUControl);

  datapath dp(clk, reset, ResultSrc, PCSrc,
              ALUSrc, RegWrite,
              ImmSrc, ALUControl,
              Zero, PC, Instr,
              ALUResult, WriteData, ReadData);
endmodule
