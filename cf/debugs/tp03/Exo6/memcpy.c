#include "memcpy.h"

int memcpy(char* src, size_t src_size, size_t src_offset, char* dest, size_t dest_size,  size_t dest_offset, size_t size);


int main(void){
	char src[17] = "The expert knows.";
	char dest[22] = "I am a Frama-C newbie!";
	int res = memcpy(src,17,14,dest,22,1,6);
	//@ assert res == -1;
	res = memcpy(src,17,4,dest,22,21,6);
	//@ assert res == -1;
	res = memcpy (src,17,19,dest,22,15,6);
	//@ assert res == -1;
	res = memcpy (src,17,4,dest,22,25,6);
	//@ assert res == -1;
	res = memcpy (src,17,4,dest,22,15,0);
	//@ assert res == 0;
	res = memcpy(src,17,4,dest,22,15,6);
	//@ assert res == 0;
	//@ assert \forall size_t i; 0<= i< 6 ==> dest[15+i] == src[4+i];
	return 0;
}