.PHONY: clean data lint requirements jupyter

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# BASE RULES                                                                    #
#################################################################################

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		conda env create -f requirements.yml
else
	@echo ">>> Conda is not installed"
	@echo ">>> Please install conda before setting this project"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) src/tests/test_environment.py

## Install Python Dependencies
requirements: test_environment
	conda env update -f requirements.yml

## Update requirements file
requirements_file: test_environment
	conda env export --from-history | grep -v "prefix" > requirements.yml

## Delete all compiled files, models and data
clean: 
	@echo "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf models/*
	rm -rf data/processed/*
	@echo "Clean also raw data ? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf data/raw/*

## Delete all compiled Python files
clean-pyc:
	@echo "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete all data
clean-data:
	@echo "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf data/processed/*
	@echo "Clean also raw data ? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf data/raw/*

## Delete all models
clean-model:
	@echo "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf models/*

## Lint using flake8
lint: test_environment
	python -m flake8 src

## Start Jupyter Notebook local server
jupyter: test_environment
	jupyter-notebook --ip=0.0.0.0

## Start Flask server
flask: test_environment
	export FLASK_APP=index
	export FLASK_ENV=development
	flask --app deployment/www/index.py run

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')