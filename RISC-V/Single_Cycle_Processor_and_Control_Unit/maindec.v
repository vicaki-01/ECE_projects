module maindec(op,
               ResultSrc,
               MemWrite,
               Branch, ALUSrc,
               RegWrite, Jump,
               ImmSrc,
               ALUOp);

  input [6:0] op;
  output [1:0] ResultSrc;
  output MemWrite;
  output Branch, ALUSrc;
  output RegWrite, Jump;
  output [1:0] ImmSrc;
  output [1:0] ALUOp;

  reg [10:0] controls;

  assign {RegWrite, ImmSrc, ALUSrc, MemWrite,
          ResultSrc, Branch, ALUOp, Jump} = controls;
  //assign PCSrc = (Btaken == 1'b1 || Jump == 1'b1) ? 2'b01 : 2'b00;

always @(*) begin
    case (op)
        7'b000_0011: controls = 14'b1_00_1_0_01_0_00_0; //lw
        7'b010_0011: controls = 14'b0_01_1_1_00_0_00_0; // sw
        7'b011_0011: controls = 14'b1_00_0_0_00_0_10_0; // R-type
        7'b110_0011: controls = 14'b0_10_0_0_00_1_01_0; // beq 0
        7'b001_0011: controls = 14'b1_00_1_0_00_0_10_0; // I-type
        7'b110_1111: controls = 14'b1_11_0_0_10_0_00_1; // jal
            default: controls = 14'b0_00_0_0_00_0_00_0; // default
        
    endcase
end

endmodule
