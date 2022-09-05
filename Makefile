CC      := g++
SRCS    := $(shell find . -name '*.cpp')
OBJS    := $(patsubst %.cpp,%.o,$(SRCS))
EXECUTABLE :=$(patsubst %.cpp,%,$(SRCS))


CFLAGS  := -I$(INCLUDE) 
LDLIBS  := -lgeogram -lexploragram

all: $(EXECUTABLE)

%.o: %.cpp
	$(CC) $(CFLAGS) $^ -c $@

$(EXECUTABLE): % : %.o
	$(CC)  $^ -o $@ $(LDLIBS)
	

