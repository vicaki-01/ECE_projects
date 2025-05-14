module regfile (clk, we3, r1, r2, r3, dw3, rd1, rd2);
    output [63:0] rd1;
    output [63:0] rd2;
    input [63:0] dw3;
    input [4:0] r1;
    input [4:0] r2;
    input [4:0] r3;
    input we3;
    input clk;
    reg [63:0] registers [31:1];
    
  // Three ported register file
  // read two ports combinationally (A1/RD1, A2/RD2)
  // write third port on rising edge of clock (A3/WD3/WE3)
  // register 0 hardwired to 0

  always @(posedge clk)
    if (we3) registers[r3] <= dw3;	

  assign rd1 = (r1 != 0) ? registers[r1] : 0;
  assign rd2 = (r2 != 0) ? registers[r2] : 0;
endmodule
