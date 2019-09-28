#ifndef _BARCODE
#define _BARCODE

struct barCode{
 unsigned int size;
 unsigned int * code;
};

struct barCode generateBarCode(unsigned int size);
void freeBarCode(struct barCode b);
void printLine(struct barCode b);
void printBarCode(struct barCode b, unsigned int h);
#endif
