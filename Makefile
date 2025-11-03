all: build

configure:
	./configure.sh

build:
	@cmake --build build

clean:
	rm -rf build

.PHONY: configure build clean
