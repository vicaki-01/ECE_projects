module extend(instr,
	        immsrc,
              immext);

  input [31:7] instr;
  input [1:0]  immsrc;
  output reg [63:0] immext;

  always @(*)
    case(immsrc)
               // I-type 
      2'b00:   immext = {{52{instr[31]}}, instr[31:20]};  
               // S-type (stores)
      2'b01:   immext = {{52{instr[31]}}, instr[31:25], instr[11:7]};
               // SB-type (branches)
      2'b10:   immext = {{52{instr[31]}}, instr[7], instr[30:25], instr[11:8], 1'b0};
               // UJ-type (jal)
      2'b11:   immext = {{44{instr[31]}}, instr[19:12], instr[20], instr[30:21], 1'b0};
      default: immext = 64'bx; // undefined
    endcase             
endmodule

