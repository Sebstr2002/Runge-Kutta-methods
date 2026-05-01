.PHONY: all build install clean test

all: build

build:
	@echo "Configuring and building with CMAKE"
	cmake -S . -B build
	cmake --build build -j

install: build
	@echo "Installing module to python/ directory..."
	mkdir -p python
	cp build/hamsolver.so python/sem/
	@echo "Done! You can now 'import hamsolver' from the python/ directory."

test: build
	@echo "Running C++ tests..."
	cd build && ctest --output-on-failure

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build
	rm -f python/hamsolver.so
