CC = nvcc
NVCC_FLAGS = -O3 -g

TARGET = q2

q2: stencil.cu
	$(CC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
