CC = nvcc
CFLAGS = -Wall

TARGET = q2

q2: stencil.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)