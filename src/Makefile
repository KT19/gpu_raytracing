NVCC = nvcc
LDFLAGS = -lGL -lglfw 
OPT=

TARGET = main
SRCS = main.cu

OBJS = $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LDFLAGS) $(OPT) -o $(TARGET) $(OBJS)

%.o: %.cu
	$(NVCC) $(LDFLAGS) -c $< 

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean