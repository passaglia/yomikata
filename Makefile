# Makefile
SHELL = /bin/bash

.PHONY: help
help:
    @echo "Commands:"
    @echo "venv    : creates a virtual environment."
    @echo "style   : executes style formatting."
    @echo "app     : runs the streamlit app."
    @echo "build   : builds the package."

# Styling
.PHONY: style
style:
	source venv/bin/activate && \
	black . && \
	isort . && \
	flake8 

# Environment 
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install pip setuptools wheel && \
	python3 -m pip install -e .

# Streamlit app
app:
	source venv/bin/activate && \
	streamlit run app.py --server.fileWatcherType none

# Build
build:
	python3 -m build