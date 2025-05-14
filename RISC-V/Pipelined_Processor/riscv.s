# Instructions.s
# ehsan.rohani@uvu.edu
# 7 Nov. 2022
#
# Test the Piplined RISC-V processor.  
# add, sub, and, or, slt, addi, ld, sd 
# If successful, it should write the value 7 to address 104 and value 8 to address 112 and ...
# value 32 to address 120 and value 8 to address 128 and value 11 to address 136
# The memory addressess are in Byte while for actual processor it is in double word
# Address:Machine Code     RISC-V Assembly         Description               
   0:	0280b503          	ld	x10,40(x1)		# Register[10]= 64'd7;
   4:	403105b3          	sub	x11,x2,x3		# Register[11]= 64'd8;
   8:	00418633          	add	x12,x3,x4		# Register[12]= 64'd32;
   c:	0300b683          	ld	x13,48(x1)		# Register[13]= 64'd8;
  10:	00628733          	add	x14,x5,x6		# Register[14]= 64'd11;
  14:	06a03423          	sd	x10,104(x0) 		# mem1[104]= Register[10]= 64'd7;
  18:	06b03823          	sd	x11,112(x0) 		# mem1[112]= Register[11]= 64'd8;
  1c:	06c03c23          	sd	x12,120(x0) 		# mem1[120]= Register[12]= 64'd32;
  20:	08d03023          	sd	x13,128(x0)  		# mem1[128]= Register[13]= 64'd8;	
  24:	08e03423          	sd	x14,136(x0) 		# mem1[136]= Register[14]= 64'd11;
  28:	00000013          	nop
  2c:	00000013          	nop
  30:	00000013          	nop
  34:	00000013          	nop
  38:	00000013          	nop
  3c:	00000013          	nop
# Make sure to sart the PC from 0x40
# If successful, it should write the value 0x1010 to address 88 and the value 4 to address 96 and
# the value -10 to address 104 and the value -24 to address 112
  40:	40308133          	sub	x2,x1,x3 		# Register[2]= 64'd-12=64'hfffffffffff4;
  44:	00517633          	and	x12,x2,x5		# Register[12]= 64'd4;
  48:	002366b3          	or	x13,x6,x2		# Register[13]= 64'd-10=64'hfffffffffff6;
  4c:	00210733          	add	x14,x2,x2		# Register[14]= 64'd-24=64'hffffffffffe8;
  50:	06f13223          	sd	x15,100(x2)		# mem1[88]= Register[15]= 64'h1010;
  54:	06c13623          	sd	x12,108(x2)		# mem1[96]= Register[12]= 64'd4;
  58:	06d13a23          	sd	x13,116(x2)		# mem1[104]= Register[13]= 64'hfffffffffff6;
  5c:	06e13e23          	sd	x14,124(x2)		# mem1[112]= Register[14]= 64'hffffffffffe8;
  60:	00000013          	nop
  64:	00000013          	nop
  68:	00000013          	nop
  6c:	00000013          	nop
# Make sure to sart the PC from 0x70
# If successful, it should write the value 5 to address 104 and the value 5 to address 112 and
# the value 7 to address 120 and the value 10 to address 128 and the value 5 to address 136
  70:	0180b103          	ld	x2,24(x1)		# Register[2]= mem1['d40]= 'd5;
  74:	00517233          	and	x4,x2,x5		# Register[4]= 'd5;
  78:	00616433          	or	x8,x2,x6		# Register[8]= 'd7;
  7c:	002204b3          	add	x9,x4,x2		# Register[9]= 'd10;
  80:	407300b3          	sub	x1,x6,x7		# Register[1]= 'd5;
  84:	06203423          	sd	x2,104(x0) 		# mem1[104]= Register[2]= 64'd5;
  88:	06403823          	sd	x4,112(x0) 		# mem1[112]= Register[4]= 64'd5; 
  8c:	06803c23          	sd	x8,120(x0) 		# mem1[120]= Register[8]= 64'd7; 
  90:	08903023          	sd	x9,128(x0) 		# mem1[128]= Register[9]= 64'd10; 
  94:	08103423          	sd	x1,136(x0) 		# mem1[136]= Register[1]= 64'd5; 
  98:	00000013          	nop
  9c:	00000013          	nop
  a0:	00000013          	nop
  a4:	00000013          	nop
  a8:	00000013          	nop
