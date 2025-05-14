module hazardunit
(input MemRead, input [4:0] RS1,RS2,RD,
output reg controlmux,PCen,IFIDen);

always@(*)
	begin
	if(MemRead)
		begin
		if((RS1 == RD) || (RS2 == RD))
		begin
			controlmux = 1'b0;
			PCen = 1'b0;
			IFIDen = 1'b0;
		end
		end

		else 
		begin
			controlmux = 1'b1;
			PCen = 1'b1;
			IFIDen = 1'b1;
		end			

	end

endmodule

