CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -Iheader
LDFLAGS = -lm
TARGET = bin/main

SRCS = main.c src/neurone.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS) | bin
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

bin:
	mkdir -p bin

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean bin
