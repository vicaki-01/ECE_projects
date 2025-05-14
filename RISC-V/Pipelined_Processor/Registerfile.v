module Registerfile (
input clk, RegWrite, input [4:0] R1, input [4:0] R2, input [4:0] W1, input [63:0] WD1,
output  [63:0] RD1, RD2);
reg [63:0] Register [31:1];
    initial begin
        Register[31] = 64'b0;
        Register[1] = 64'd16;
        Register[2] = 64'd36;
        Register[3] = 64'd28;
        Register[4] = 64'd4;
        Register[5] = 64'd5;
        Register[6] = 64'd6;
        Register[7] = 64'd1;
        Register[15]=64'd4112;
    end
always @(negedge clk)
begin
if(RegWrite && W1!=0)
Register[W1] <= WD1;
end
assign RD1 = (R1 != 0) ? Register[R1] : 0;
assign RD2 = (R2 != 0) ? Register[R2] : 0;
endmodule

