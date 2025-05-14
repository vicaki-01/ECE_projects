module EXU (input [63:0] PC, RD1, RD2, Immediatevalue, 
input [3:0] Alucontrolinput, 
input [8:0] controlsignal,
input [4:0] RS1,RS2,Rd2,Rd1,
input [63:0] ALUIN,Writedatain,
output zero, 
output [63:0] ALUresult, 
output [63:0] branchaddress  
);

wire [1:0] ForwardA,ForwardB;
wire [63:0] immediateout,ALUinput1,MUXout,ALUinput2;
wire [3:0] Alucontrol;
forwarding Forwarding(controlsignal[6],RS1,RS2,Rd1,Rd2,ForwardA,ForwardB);
shift Shift(Immediatevalue,immediateout);
add Add(PC,immediateout,branchaddress);
MUX1 Mux5(RD1,Writedatain,ALUIN,ForwardA,ALUinput1);
MUX1 Mux6(RD2,Writedatain,ALUIN,ForwardB,MUXout);
MUX Mux2(MUXout,Immediatevalue,controlsignal[8],ALUinput2);
ALUControl ALU_Control(controlsignal[1:0],Alucontrolinput[3],Alucontrolinput[2:0],Alucontrol);
ALU ALU_Unit(Alucontrol,ALUinput1,ALUinput2,ALUresult,zero);


endmodule  
