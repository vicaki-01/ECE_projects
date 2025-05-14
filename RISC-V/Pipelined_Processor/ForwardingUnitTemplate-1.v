module forwarding(
input RegWrite, input [4:0] RS1, RS2, RD1, RD2,
output reg [1:0] ForwardA , ForwardB);

always@(*)
	begin
	
	if (RegWrite)
	begin
	if(RS1 == RD1 && RS1 != 0)
		ForwardA = 2'b10;
	else if (RS1 == RD2 && RS1 != 0)
		ForwardA = 2'b01;
	else
		ForwardA = 2'b00;
	
	if(RS2 == RD1 && RS1 != 0)
		ForwardB = 2'b10;
	else if(RS2 == RD2 && RS1 != 0)
		ForwardB = 2'b01;
	else
		ForwardB = 2'b00;
	end
	end


endmodule
