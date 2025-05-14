module Pipeline (clk, reset, initialPC,
           WriteData, DataAdr, 
           MemWrite);

input  clk, reset; 
input [63:0] initialPC;
output [63:0] WriteData, DataAdr; 
output  MemWrite;

wire [63:0] PC, N6,RD1,RD2,Immediate, ALUout,BranchTargetAdd,MemData,PCPlus4;
wire [31:0] Instruction;
wire [4:0] W3; // destination reg
wire [3:0] Func; // {funct7b5, funct3}
wire [8:0] Controls; // control signals
wire Zero; // zero bit
// wire [4:0] N14; // destination reg
// wire [5:0] N15; // Control Signal
wire AddSel,N22;
wire [63:0] N23;
wire [4:0] N24;
wire [2:0] N19; // Control Signal
// wire [4:0] N20; // destination reg
wire [4:0] RS1, RS2; // RS1 and RS2 field 
reg [31:0] IF2ID_Instruction;
reg [63:0] IF2ID_PC;
reg [63:0] ID2EX_PC, ID2EX_RD1, ID2EX_RD2, ID2EX_Immediate;
reg [4:0] ID2EX_W3;
reg [3:0] ID2EX_Func;
reg [8:0] ID2EX_Controls;
reg [63:0] EX2MeM_ALUout,EX2MeM_BranchTargetAdd,EX2MeM_RD2;
reg EX2MeM_Zero; // zero bit
reg [4:0] EX2MeM_W3; // destination reg
reg [5:0] EX2MeM_Controls; // control signal
reg [63:0] MeM2WB_BranchTargetAdd, MeM2WB_MemData, MeM2WB_ALUout;
reg [2:0] MeM2WB_Controls; // controlsignal
reg [4:0] MeM2WB_W3; // destination reg
reg MeM2WB_AddSel;
wire RegWrite;
wire [63:0] WriteBack;
wire [4:0] Rd;
reg [4:0] ID2EX_RS1;
reg [4:0] ID2EX_RS2;
wire PCen,FetchNew;
wire Mem2Reg;
reg ID2EX_PCen, IF2ID_FetchNew;

// The Pipeline Stage Registers

always @(posedge clk)
begin
if(reset==1)
begin
// IF2ID Pipeline Registers
// Needs to be reseted to make sure it does not create hazard in ID Stage
// Either this or we should reset the output of hazard unit in the second cycle
	IF2ID_Instruction <= 32'h00000013;
	IF2ID_PC <= initialPC;
end
else if(FetchNew == 0)
begin
	IF2ID_Instruction <= IF2ID_Instruction;
	IF2ID_PC <= IF2ID_PC;
end
else
begin
	IF2ID_Instruction <= Instruction; // Instruction
	IF2ID_PC <= PC; // PC
end


if (reset==1) begin
	IF2ID_FetchNew <= 1; // Fetch New Instruction
end
else begin
	IF2ID_FetchNew <= FetchNew; // Fetch New Instruction
end
// ID2EX Pipline Registers
ID2EX_PC <= IF2ID_PC; // PC
ID2EX_RD1 <= RD1; // Read Data 1 (RD1)
ID2EX_RD2 <= RD2; // Read Data 2 (RD2)
ID2EX_Immediate <= Immediate; // Immediate value 

if (reset==1) begin 
	ID2EX_W3 <= 0; // Destination Reg (W3)
end
else begin
	ID2EX_W3 <= W3; // Destination Reg (W3)
end
ID2EX_Func <= Func; // {funct7b5, funct3}

if (reset==1) begin
	ID2EX_Controls <= 0;
end
else begin
	ID2EX_Controls <= Controls; // {ALUSrc,MemtoReg,Regwrite,MemRead,MemWrite,Branch,jump,ALUop} -> Control Signals
end

ID2EX_RS1 <= RS1; // RS1
ID2EX_RS2 <= RS2; //RS2

// EX2MeM Pipline Registers
EX2MeM_ALUout <= ALUout; // ALU Result 
EX2MeM_BranchTargetAdd <= BranchTargetAdd; // Branch Target Address
EX2MeM_RD2 <= ID2EX_RD2; // RD2
EX2MeM_Zero <= Zero; // zero flag from ALU Output

if (reset==1) begin
	EX2MeM_W3 <= 0; //Destination reg
end
else begin
	EX2MeM_W3 <= ID2EX_W3; //Destination reg
end

if (reset==1) begin
	EX2MeM_Controls <= 0; // control signals
end
else begin
	EX2MeM_Controls <= ID2EX_Controls[7:2]; // control signals
end
// MeM2WB Pipline Registers
MeM2WB_BranchTargetAdd <= EX2MeM_BranchTargetAdd; // Branch Target Address
MeM2WB_MemData <= MemData; // Memory Data
MeM2WB_ALUout <= EX2MeM_ALUout; // ALU Result

if (reset==1) begin
	MeM2WB_Controls <= 0; // conrol signal
end
else begin
	MeM2WB_Controls <= N19; // conrol signal
end

MeM2WB_W3 <= EX2MeM_W3; // Destination reg
// To  make sure The pipeline continues while pipeline is not filled
if (reset==1) begin
	MeM2WB_AddSel <= 0; // Address Select
end
else begin
	MeM2WB_AddSel <= AddSel;
end
	
ID2EX_PCen <= PCen;
end


IFU IF(clk,reset,initialPC,MeM2WB_AddSel,PCen,PCPlus4,EX2MeM_BranchTargetAdd,PC,Instruction);
IDU ID(clk, RegWrite,IF2ID_Instruction,WriteBack,Rd,ID2EX_W3,ID2EX_Controls[5],RD1,RD2,Immediate,W3,Func,Controls,RS1,RS2,PCen,FetchNew);
EXU EX(ID2EX_PC,ID2EX_RD1,ID2EX_RD2,ID2EX_Immediate,ID2EX_Func,ID2EX_Controls,ID2EX_RS1,ID2EX_RS2,MeM2WB_W3,EX2MeM_W3,EX2MeM_ALUout,WriteBack,Zero,ALUout,BranchTargetAdd);
MeMU MeM(clk,EX2MeM_ALUout,EX2MeM_RD2,EX2MeM_Zero,EX2MeM_Controls,MemData,AddSel);
WBU WB(MeM2WB_MemData,MeM2WB_ALUout,Mem2Reg,WriteBack);



assign Mem2Reg=MeM2WB_Controls[2];

assign N19 = {EX2MeM_Controls[5:4],EX2MeM_Controls[0]};
assign RegWrite=MeM2WB_Controls[1];
assign Rd=MeM2WB_W3;// Register Destination of Instruction in WB Stage
assign WriteData=EX2MeM_RD2;

// Addes for the testbench
assign DataAdr=EX2MeM_ALUout;
assign MemWrite=EX2MeM_Controls[2];



//
//
//
endmodule

