//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------*Top View of Processor with Memories*-----------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//----------------Three Seprate Modules:---------------------
//------Processor, Data Memory and Instruction Memory--------
//----Memory Elements Are Normaly Synthesized Seperately-----  

module top(clk, reset, 
           WriteData, DataAdr, 
           MemWrite);

  input  clk, reset; 
  output [63:0] WriteData, DataAdr; 
  output  MemWrite;

  wire [63:0] PC, ReadData;
  wire [31:0] Instr;

  // instantiate processor and memories
  riscvsingle rvsingle(clk, reset, PC, Instr, MemWrite, DataAdr, 
                       WriteData, ReadData);
  imem imem(PC, Instr);
  dmem dmem(clk, MemWrite, DataAdr, WriteData, ReadData);
endmodule
