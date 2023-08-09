######################################################################
# @author      : t-hara (t-hara@$HOSTNAME)
# @file        : Makefile
# @created     : Friday Oct 21, 2022 23:55:49 JST
######################################################################

IDIR =./include
CC=gcc
CFLAGS=-I$(IDIR)
ODIR=.

main:
	$(CC) -Wall -fPIC -c mlp_params.c mlp.c nn_math.c nn.c
	$(CC) -shared mlp_params.o mlp.o nn_math.o nn.o -o mlp.so

mlp:
	# $(CC) -o $@ $^ $(CFLAGS) $(LIBS)
	$(CC) -Wall -fPIC -c mlp_params.c mlp.c nn_math.c nn.c
	$(CC) -shared mlp_params.o mlp.o nn_math.o nn.o -o mlp.so

.PHONY: clean

clean:
	rm -f $(ODIR)/*.{o,so}

