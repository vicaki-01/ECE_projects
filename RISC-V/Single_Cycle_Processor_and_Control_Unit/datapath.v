//-----------------------------------------------------------
//-----------------------------------------------------------
//------------------* Datapath Elements *--------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------

module datapath(clk, reset,
                ResultSrc, 
                PCSrc, ALUSrc,
                RegWrite,
                ImmSrc,
                ALUControl,
                Zero,
                PC,
                Instr,
                ALUResult, WriteData,
                ReadData);

  input  clk, reset;
  input  [1:0] ResultSrc; 
  input  PCSrc, ALUSrc;
  input  RegWrite;
  input  [1:0] ImmSrc;
  input  [2:0]  ALUControl;
  output Zero;
  output [63:0]  PC;
  input  [31:0] Instr;
  output [63:0] ALUResult, WriteData;
  input  [63:0] ReadData;

  wire [63:0] PCNext, PCPlus4;
  wire [63:0] ImmExt;
  wire [63:0] SrcA, SrcB;
  wire [63:0] Result;

  //-----Program Counter
  // PC Register
  flopr pcreg(clk, reset, PCNext, PC);	
  
  // Initiate Next PC Logic Here
  NextPCLogic pcNEXT(PCNext, PCPlus4, PC, ImmExt, PCSrc);


  // register file logic
  // Initiate the regfile module here
  regfile regFile(clk, RegWrite, Instr[19:15], Instr[24:20], Instr[11:7], Result, SrcA, WriteData);

  
  // Sign Enderder Logid
  // Initiate the extended module here
  extend ext(Instr[31:7], ImmSrc, ImmExt[63:0]);

  // ALU logic
  mux2  srcbmux(WriteData, ImmExt, ALUSrc, SrcB);
  
  // Initiate the alu module here
  alu aluUnit(SrcA, SrcB, ALUControl, ALUResult, Zero);

  mux3  resultmux(ALUResult, ReadData, PCPlus4, ResultSrc, Result);
endmodule

module flopr  (clk, reset, d, q);

  input  clk, reset;
  input  [63:0] d; 
  output reg [63:0] q;


  always @(posedge clk, posedge reset)
    if (reset) q <= 0;
    else       q <= d;
endmodule

module mux2  (d0, d1, s, y);
	input  [63:0] d0, d1; 
        input  s; 
        output [63:0] y;

  assign y = s ? d1 : d0; 
endmodule

module mux3 (d0, d1, d2, s, y);

  input  [63:0] d0, d1, d2;
  input  [1:0] s;
  output [63:0] y;

  assign y = s[1] ? d2 : (s[0] ? d1 : d0); 
endmodule



