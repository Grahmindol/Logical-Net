CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11 -Iheader
LDFLAGS = -lm

TARGET = bin/main
OBJDIR = bin/obj

SRCS = main.c src/neurone.c src/layer.c src/network.c
OBJS = $(OBJDIR)/main.o $(OBJDIR)/neurone.o $(OBJDIR)/layer.o $(OBJDIR)/network.o

all: $(TARGET)

$(TARGET): $(OBJS) | bin $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/main.o: main.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/neurone.o: src/neurone.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/layer.o: src/layer.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/network.o: src/network.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

bin:
	mkdir -p bin

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET)

.PHONY: all clean bin
