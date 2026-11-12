CC = gcc
CFLAGS = -O2 -Wall $(shell pkg-config --cflags sdl2)
LIBS = -lOpenCL $(shell pkg-config --libs sdl2) -lm

galaxia: galaxia.c
	$(CC) $(CFLAGS) -o galaxia galaxia.c $(LIBS)

run: galaxia
	./galaxia

clean:
	rm -f galaxia captura.ppm
