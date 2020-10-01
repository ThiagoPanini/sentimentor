.PHONY: help clean test clean-build isort run

.DEFAULT: help

help:
	@echo "make clean"
	@echo "       prepare development environment, use only once"
	@echo "make clean-build"
	@echo "       Clear all build directories"
	@echo "make env"
	@echo "       Create env"
	@echo "make active"
	@echo "       Activate virtual environment and install packages"
	@echo "make deactivate"
	@echo "       Deactivate and remove virtual environment"
	@echo "make run"
	@echo "       run the web application"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . | grep -E "__pycache__|.pyc|.DS_Store$$" | xargs rm -rf

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

env:
	python -m venv env/
	pip install -r requirements.txt

test:
	pytest --verbose --color=yes

run:
	python dev/api/main.py