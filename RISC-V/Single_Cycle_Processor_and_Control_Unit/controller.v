//-----------------------------------------------------------
//-----------------------------------------------------------
//-------------------* Control Module*-----------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-------The Two Control Modules Put together to Create------
//--------------------the Control Unit-----------------------  


module controller(op,
                  funct3,
                  funct7b5,
                  Zero,
                  ResultSrc,
                  MemWrite,
                  PCSrc, ALUSrc,
                  RegWrite, Jump,
                  ImmSrc,
                  ALUControl);

  input  [6:0] op;
  input  [2:0] funct3;
  input  funct7b5;
  input  Zero;
  output [1:0] ResultSrc;
  output MemWrite;
  output PCSrc, ALUSrc;
  output RegWrite, Jump;
  output [1:0] ImmSrc;
  output [2:0] ALUControl;


  wire [1:0] ALUOp;
  wire       Branch;

  maindec md(op, ResultSrc, MemWrite, Branch,
             ALUSrc, RegWrite, Jump, ImmSrc, ALUOp);
  aludec  ad(op[5], funct3, funct7b5, ALUOp, ALUControl);

  assign PCSrc = Branch & Zero | Jump;
endmodule
