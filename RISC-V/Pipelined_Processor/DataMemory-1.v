//-----------------------------------------------------------
//-----------------------------------------------------------
//---------------------*Memory Elements*---------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//----------------Instruction Memory Is Loaded---------------
//---with Data Specified in riscvtest.txt Using $readmemh----
module DataMemory (input clk, MemWrite, MemRead, 
input [63:0] address, WriteData,
output reg [63:0] ReadData);
reg [63:0] mem1[255:0];
// Fro simplicity the memory is implemented as Double Word
// addressable and not byte addressable
    initial begin
        mem1[0]  = 64'd0;
        mem1[1]  = 64'd1;
        mem1[2] = 64'd2;
        mem1[3] = 64'd3;
        mem1[4] = 64'd4;
        mem1[5] = 64'd5;
        mem1[6] = 64'd6;
        mem1[7] = 64'd7;
        mem1[8] = 64'd8;
        mem1[9] = 64'd9;
        mem1[10] = 64'd10;
        mem1[11] = 64'd11;
        mem1[12] = 64'd12;
    end
always @(*)
begin
if(MemRead == 1'b1)
ReadData <= mem1[address>>3];
else if(MemWrite == 1'b1)
mem1[address>>3] = WriteData;
end
endmodule
