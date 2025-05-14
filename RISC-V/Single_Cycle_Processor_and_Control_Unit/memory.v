//-----------------------------------------------------------
//-----------------------------------------------------------
//---------------------*Memory Elements*---------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//----------------Instruction Memory Is Loaded---------------
//---with Data Specified in riscvtest.txt Using $readmemh----

module imem(a, rd);

	input  [63:0] a;
        output [31:0] rd;

  reg [31:0] RAM[63:0];

  initial
      $readmemh("riscvtest.txt",RAM);

  assign rd = RAM[a[31:2]]; // word aligned
endmodule

module dmem(clk, we, a, wd, rd);

	input  clk, we;
        input  [63:0] a, wd;
        output [63:0] rd;

  reg [63:0] RAM[255:0];

  assign rd = RAM[a[63:3]]; // word aligned

  always @(posedge clk)
    if (we) RAM[a[63:3]] <= wd;
endmodule
