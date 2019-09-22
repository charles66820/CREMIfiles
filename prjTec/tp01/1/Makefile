all : main
	./helloWorld

main.o : main.c
	gcc -c main.c

libfoobar.a : foo.o bar.o
	ar -cr libfoobar.a foo.o bar.o

bar.o : bar.c bar.h
	gcc -c bar.c

foo.o : foo.c foo.h
	gcc -c foo.c

main : main.o libfoobar.a
	gcc main.o -lm -L. -lfoobar -o helloWorld

clean :
	rm -f *.o *.a
