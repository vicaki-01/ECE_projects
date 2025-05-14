module PCCounter (input clk, reset, input [63:0] initialPC, 
input PCen, input [63:0] PC1, output reg [63:0] PC);

always@(posedge clk)
begin
if (reset==1)
PC<=initialPC;
else if(PCen == 0)
PC <= PC;
else 
PC <= PC1;
end
endmodule
