
makethreads.coff:     file format ecoff-littlemips


Disassembly of section .text:

00000000 <_ftext>:
   0:	ffffffff 	0xffffffff
   4:	ffffffff 	0xffffffff
	...

00000080 <__start>:
  80:	0c000097 	jal	25c <main>
  84:	00000000 	nop
  88:	0c00004a 	jal	128 <Exit>
  8c:	00402021 	move	r4,r2
  90:	ffffffff 	0xffffffff
  94:	ffffffff 	0xffffffff
	...

00000100 <__thread>:
 100:	0040f809 	jalr	r2
 104:	00000000 	nop
 108:	0c000086 	jal	218 <ThreadExit>
 10c:	00402021 	move	r4,r2
 110:	ffffffff 	0xffffffff
 114:	ffffffff 	0xffffffff

00000118 <Halt>:
 118:	24020000 	li	r2,0
 11c:	0000000c 	syscall
 120:	03e00008 	jr	r31
 124:	00000000 	nop

00000128 <Exit>:
 128:	24020001 	li	r2,1
 12c:	0000000c 	syscall
 130:	03e00008 	jr	r31
 134:	00000000 	nop

00000138 <Exec>:
 138:	24020002 	li	r2,2
 13c:	0000000c 	syscall
 140:	03e00008 	jr	r31
 144:	00000000 	nop

00000148 <Join>:
 148:	24020003 	li	r2,3
 14c:	0000000c 	syscall
 150:	03e00008 	jr	r31
 154:	00000000 	nop

00000158 <Create>:
 158:	24020004 	li	r2,4
 15c:	0000000c 	syscall
 160:	03e00008 	jr	r31
 164:	00000000 	nop

00000168 <Open>:
 168:	24020005 	li	r2,5
 16c:	0000000c 	syscall
 170:	03e00008 	jr	r31
 174:	00000000 	nop

00000178 <Read>:
 178:	24020006 	li	r2,6
 17c:	0000000c 	syscall
 180:	03e00008 	jr	r31
 184:	00000000 	nop

00000188 <Write>:
 188:	24020007 	li	r2,7
 18c:	0000000c 	syscall
 190:	03e00008 	jr	r31
 194:	00000000 	nop

00000198 <Close>:
 198:	24020008 	li	r2,8
 19c:	0000000c 	syscall
 1a0:	03e00008 	jr	r31
 1a4:	00000000 	nop

000001a8 <Fork>:
 1a8:	24020009 	li	r2,9
 1ac:	0000000c 	syscall
 1b0:	03e00008 	jr	r31
 1b4:	00000000 	nop

000001b8 <Yield>:
 1b8:	2402000a 	li	r2,10
 1bc:	0000000c 	syscall
 1c0:	03e00008 	jr	r31
 1c4:	00000000 	nop

000001c8 <PutChar>:
 1c8:	2402000b 	li	r2,11
 1cc:	0000000c 	syscall
 1d0:	03e00008 	jr	r31
 1d4:	00000000 	nop

000001d8 <PutString>:
 1d8:	2402000c 	li	r2,12
 1dc:	0000000c 	syscall
 1e0:	03e00008 	jr	r31
 1e4:	00000000 	nop

000001e8 <GetChar>:
 1e8:	2402000d 	li	r2,13
 1ec:	0000000c 	syscall
 1f0:	03e00008 	jr	r31
 1f4:	00000000 	nop

000001f8 <GetString>:
 1f8:	2402000e 	li	r2,14
 1fc:	0000000c 	syscall
 200:	03e00008 	jr	r31
 204:	00000000 	nop

00000208 <ThreadCreate>:
 208:	24020011 	li	r2,17
 20c:	0000000c 	syscall
 210:	03e00008 	jr	r31
 214:	00000000 	nop

00000218 <ThreadExit>:
 218:	24020012 	li	r2,18
 21c:	0000000c 	syscall
 220:	03e00008 	jr	r31
 224:	00000000 	nop

00000228 <__main>:
 228:	03e00008 	jr	r31
 22c:	00000000 	nop

00000230 <f>:
 230:	27bdffe8 	addiu	r29,r29,-24
 234:	afbf0010 	sw	r31,16(r29)
 238:	80840000 	lb	r4,0(r4)
 23c:	0c000072 	jal	1c8 <PutChar>
 240:	00000000 	nop
 244:	0c000072 	jal	1c8 <PutChar>
 248:	2404000a 	li	r4,10
 24c:	8fbf0010 	lw	r31,16(r29)
 250:	00000000 	nop
 254:	03e00008 	jr	r31
 258:	27bd0018 	addiu	r29,r29,24

0000025c <main>:
 25c:	3c040000 	lui	r4,0x0
 260:	27bdffe0 	addiu	r29,r29,-32
 264:	24020061 	li	r2,97
 268:	24840300 	addiu	r4,r4,768
 26c:	afbf0018 	sw	r31,24(r29)
 270:	0c000076 	jal	1d8 <PutString>
 274:	a3a20010 	sb	r2,16(r29)
 278:	3c040000 	lui	r4,0x0
 27c:	27a50010 	addiu	r5,r29,16
 280:	0c000082 	jal	208 <ThreadCreate>
 284:	24840230 	addiu	r4,r4,560
 288:	3c040000 	lui	r4,0x0
 28c:	0c000076 	jal	1d8 <PutString>
 290:	24840320 	addiu	r4,r4,800
 294:	0c000086 	jal	218 <ThreadExit>
 298:	00000000 	nop
 29c:	8fbf0018 	lw	r31,24(r29)
 2a0:	00001021 	move	r2,r0
 2a4:	03e00008 	jr	r31
 2a8:	27bd0020 	addiu	r29,r29,32
	...
