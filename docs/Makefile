.PHONY: help
help:
	@echo "make cpp|py|all"

.PHONY: cpp
cpp:
	@echo "Building C++ documentation..."
	doxygen Doxyfile doc_cpp

.PHONY: py
py:
	@echo "Building Python documentation..."
	mkdir -p build && cd build && cmake ../../ -DBUILD_PYTHON_BINDINGS=ON && make -j
	sphinx-build -b singlehtml . ./doc_py/

.PHONY: mkdocs
mkdocs:
	@echo "Building MkDocs documentation..."
	cd .. && mkdocs build

.PHONY: all
all: cpp py mkdocs
	@echo "All documentation built."

.PHONY: deploy
deploy:
	@echo "Deploying documentation..."
	cd .. && mkdocs gh-deploy --force
