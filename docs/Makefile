.PHONY: help
help:
	@echo "make cpp|py|all"

.PHONY: cpp
cpp:
	@echo "Building C++ documentation..."
	doxygen Doxyfile

.PHONY: py
py:
	@echo "Building Python documentation..."
	mkdir -p build && cd build && cmake ../../ -DBUILD_PYTHON_BINDINGS=ON && make -j
	sphinx-build -b singlehtml . ./html_py/

.PHONY: all
all: cpp py