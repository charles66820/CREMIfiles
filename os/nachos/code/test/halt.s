
halt.coff:     file format ecoff-littlemips


Disassembly of section .text:

00000000 <_ftext>:
   0:	ffffffff 	0xffffffff
   4:	ffffffff 	0xffffffff
	...

00000080 <__start>:
  80:	0c000054 	jal	150 <main>
  84:	00000000 	nop
  88:	0c00002a 	jal	a8 <Exit>
  8c:	00002021 	move	r4,r0
  90:	ffffffff 	0xffffffff
  94:	ffffffff 	0xffffffff

00000098 <Halt>:
  98:	24020000 	li	r2,0
  9c:	0000000c 	syscall
  r4:	03e00008 	jr	r31
  a4:	00000000 	nop

000000a8 <Exit>:
  a8:	24020001 	li	r2,1
  ac:	0000000c 	syscall
  b0:	03e00008 	jr	r31
  b4:	00000000 	nop

000000b8 <Exec>:
  b8:	24020002 	li	r2,2
  bc:	0000000c 	syscall
  c0:	03e00008 	jr	r31
  c4:	00000000 	nop

000000c8 <Join>:
  c8:	24020003 	li	r2,3
  cc:	0000000c 	syscall
  d0:	03e00008 	jr	r31
  d4:	00000000 	nop

000000d8 <Create>:
  d8:	24020004 	li	r2,4
  dc:	0000000c 	syscall
  e0:	03e00008 	jr	r31
  e4:	00000000 	nop

000000e8 <Open>:
  e8:	24020005 	li	r2,5
  ec:	0000000c 	syscall
  f0:	03e00008 	jr	r31
  f4:	00000000 	nop

000000f8 <Read>:
  f8:	24020006 	li	r2,6
  fc:	0000000c 	syscall
 100:	03e00008 	jr	r31
 104:	00000000 	nop

00000108 <Write>:
 108:	24020007 	li	r2,7
 10c:	0000000c 	syscall
 110:	03e00008 	jr	r31
 114:	00000000 	nop

00000118 <Close>:
 118:	24020008 	li	r2,8
 11c:	0000000c 	syscall
 120:	03e00008 	jr	r31
 124:	00000000 	nop

00000128 <Fork>:
 128:	24020009 	li	r2,9
 12c:	0000000c 	syscall
 130:	03e00008 	jr	r31
 134:	00000000 	nop

00000138 <Yield>:
 138:	2402000a 	li	r2,10
 13c:	0000000c 	syscall
 140:	03e00008 	jr	r31
 144:	00000000 	nop

00000148 <__main>:
 148:	03e00008 	jr	r31
 14c:	00000000 	nop

00000150 <main>:
 150:	27bdffe8 	addiu	r29,r29,-24
 154:	afbf0010 	sw	r31,16(r29)
 158:	0c000026 	jal	98 <Halt>
 15c:	00000000 	nop
	...
