.PHONY: default clean

SOURCE	= $(wildcard *.c)
EXEC	= $(SOURCE:.c=)

CC	= gcc
CFLAGS	= -fopenmp -O3
LDFLAGS	= -fopenmp

default: $(EXEC)

clean:
	rm -f $(EXEC)
