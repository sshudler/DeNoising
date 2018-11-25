# Makefile

CC = g++
MAIN = denoise_test
HDRS = NoiseCleaner.h Utils.h
SRCS = DeNoising_1_main.cpp NoiseCleaner.cpp Utils.cpp
OBJS = $(SRCS:.cpp=.o)
CFLAGS = -I/usr/include/opencv
LIBS = -lcv -lhighgui -lOpenCL


.SUFFIXES:
.SUFFIXES: .cpp .o


all: $(MAIN)


$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) -o $(MAIN) $(OBJS) $(LIBS)


$(SRCS): $(HDRS)

.cpp.o:
	$(CC) $(CFLAGS) -c $<  -o $@


clean:
	rm -f *.o
