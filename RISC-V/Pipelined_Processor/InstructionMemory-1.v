module instructionmemory (
    input [63:0] address,
    output [63:0] RD
    );
//wire [3:0] add = address[3:0];
reg [32-1:0] mem1[511:0];

initial
    $readmemh("riscvtest.txt",mem1);


assign RD = mem1[address>>2];

endmodule
