.PHONY: default clean

SOURCE	= $(wildcard *.c)
EXEC	= $(SOURCE:.c=)

CC	= gcc
CFLAGS	= -O3

default: $(EXEC)

clean:
	rm -f $(EXEC)
