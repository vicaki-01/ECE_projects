module IDU (input clk,RegWrite, 
input [31:0] Instruction,
input [63:0] WriteData,
input [4:0] Rd,RD,
input Memread,
output [63:0] RD1,RD2,immediatevalue,
output [4:0] rd, 
output [3:0] ALUcontrolinput,
output [8:0] control, 
output [4:0] RS1,RS2,
output PC_En,IFID_En);

wire controlmux;
wire [1:0] ALUop;
wire ALUSrc,MemtoReg,Regwrite,MemRead,MemWrite,Branch,jump;



assign RS1 = Instruction[19:15];
assign RS2 = Instruction[24:20];

// RD is the register destination of the instruction in EX Stage
hazardunit HazardUnit(Memread,RS1,RS2,RD,controlmux,PC_En,IFID_En);
mainController Controller(Instruction[6:0],ALUSrc,MemtoReg,Regwrite,MemRead,MemWrite,Branch,jump,ALUop);

// Rd is the register destination of instruction in WB Stage
// RD1 and RD2 are Read Data 1 and Read Data 2 of the Register File (Instruction in ID Stage)
Registerfile RegisterFile(clk,RegWrite,RS1,RS2,Rd,WriteData,RD1,RD2);
immediategeneration SignExtender(Instruction,immediatevalue);

// When there is a buble making sure the buble does not change the register or memory values or PC
assign control = (controlmux == 1'b1) ? {ALUSrc,MemtoReg,Regwrite,MemRead,MemWrite,Branch,jump,ALUop} : 0;
assign ALUcontrolinput = {Instruction[30],Instruction[14:12]};

// to make sure hazard does not create false hazrd later
// rd is the destinatoin register of the current instruction in ID
assign rd = (controlmux == 1'b1) ? Instruction[11:7]: 0;

endmodule


