//-----------------------------------------------------------
//-----------------------------------------------------------
//--------------*Short Description of ISA*-------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------

// RISC-V single-cycle processor
// From Section 7.6 of Digital Design & Computer Architecture
// 27 April 2020
// David_Harris@hmc.edu 
// Sarah.Harris@unlv.edu
// Updateted by Ehsan.Rohani@uvu.edu on 24 Oct. 2022 for I64
// and Verilog instead of system Verilog

// run 210
// Expect simulator to print "Simulation succeeded"
// when the value 25 (0x19) is written to address 100 (0x64)

// Single-cycle implementation of RISC-V (RV32I)
// User-level Instruction Set Architecture V2.2 (May 7, 2017)
// Implements a subset of the base integer instructions:
//    lw, sw
//    add, sub, and, or, slt, 
//    addi, andi, ori, slti
//    beq
//    jal
// Exceptions, traps, and interrupts not implemented
// little-endian memory

// 31 32-bit registers x1-x31, x0 hardwired to 0
// R-Type instructions
//   add, sub, and, or, slt
//   INSTR rd, rs1, rs2
//   Instr[31:25] = funct7 (funct7b5 & opb5 = 1 for sub, 0 for others)
//   Instr[24:20] = rs2
//   Instr[19:15] = rs1
//   Instr[14:12] = funct3
//   Instr[11:7]  = rd
//   Instr[6:0]   = opcode
// I-Type Instructions
//   lw, I-type ALU (addi, andi, ori, slti)
//   lw:         INSTR rd, imm(rs1)
//   I-type ALU: INSTR rd, rs1, imm (12-bit signed)
//   Instr[31:20] = imm[11:0]
//   Instr[24:20] = rs2
//   Instr[19:15] = rs1
//   Instr[14:12] = funct3
//   Instr[11:7]  = rd
//   Instr[6:0]   = opcode
// S-Type Instruction
//   sw rs2, imm(rs1) (store rs2 into address specified by rs1 + immm)
//   Instr[31:25] = imm[11:5] (offset[11:5])
//   Instr[24:20] = rs2 (src)
//   Instr[19:15] = rs1 (base)
//   Instr[14:12] = funct3
//   Instr[11:7]  = imm[4:0]  (offset[4:0])
//   Instr[6:0]   = opcode
// B-Type Instruction
//   beq rs1, rs2, imm (PCTarget = PC + (signed imm x 2))
//   Instr[31:25] = imm[12], imm[10:5]
//   Instr[24:20] = rs2
//   Instr[19:15] = rs1
//   Instr[14:12] = funct3
//   Instr[11:7]  = imm[4:1], imm[11]
//   Instr[6:0]   = opcode
// J-Type Instruction
//   jal rd, imm  (signed imm is multiplied by 2 and added to PC, rd = PC+4)
//   Instr[31:12] = imm[20], imm[10:1], imm[11], imm[19:12]
//   Instr[11:7]  = rd
//   Instr[6:0]   = opcode

//   Instruction  opcode    funct3    funct7
//   add          0110011   000       0000000
//   sub          0110011   000       0100000
//   and          0110011   111       0000000
//   or           0110011   110       0000000
//   slt          0110011   010       0000000
//   addi         0010011   000       immediate
//   andi         0010011   111       immediate
//   ori          0010011   110       immediate
//   slti         0010011   010       immediate
//   beq          1100011   000       immediate
//   lw	          0000011   010       immediate
//   sw           0100011   010       immediate
//   jal          1101111   immediate immediate



//-----------------------------------------------------------
//-----------------------------------------------------------
//------------------* Testbench setup *----------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------
//-----------------------------------------------------------

module PipelinedProcTest();

  reg clk;
  reg reset;
  reg [63:0] initialPC;

  wire [63:0] WriteData, DataAdr;
  wire MemWrite;

  reg [2:0]test = 0;

  initial begin
  #170 $finish;
  end
  initial begin
// initialPC should be 0 for 1st Program test and
// should be 64'h40 for the 2nd program test and 
// should be 64'h70 for the 3rd program test
  initialPC=64'h00;
  test = 1;
  end
  // instantiate device to be tested
  Pipeline dut(clk, reset, initialPC, WriteData, DataAdr, MemWrite);
  
  // initialize test
  initial
    begin
      reset <= 1; # 12; reset <= 0;
    end
  // generate clock to sequence tests
  always
    begin
      clk <= 1; # 5; clk <= 0; # 5;
    end

  // check results
  always @(negedge clk)
    begin
  // I wrote the 1st check for the 1st program
  // you should change the testbech to handle the rest 
      if(MemWrite) begin
        if(DataAdr == 104 & WriteData == 7)
          $display("Simulation succeeded $d", test);

	else if(DataAdr == 112 & WriteData == 8) 
          $display("Simulation succeeded %d", test);
	else if(DataAdr == 120 & WriteData == 32)
          $display("Simulation succeeded %d", test);
	else if(DataAdr == 128 & WriteData == 8)
          $display("Simulation succeeded %d", test);
	else if(DataAdr == 136 & WriteData == 11)
          $display("Simulation succeeded %d", test);
        else
          $display("Simulation failed %d, Expected: %d, Actual: %d", test, DataAdr, WriteData);
	test = test + 1;
        end
	
    end
endmodule

