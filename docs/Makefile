.PHONY: help
help:
	@echo "make cpp|py"

.PHONY: cpp
cpp:
	@echo "Building C++ documentation..."
	doxygen Doxyfile

.PHONY: py
py:
	@echo "Building Python documentation..."
	pip wheel .. -w ./wheel
	sphinx-build -b singlehtml . ./html_py/