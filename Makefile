SOURCE_DIR := src/splendor
DOCS_SOURCE_DIR := docs/source
DOCS_BUILD_DIR := docs/_build/html
COMMIT_ID := $(shell git rev-parse HEAD)
COMMIT_MSG := $(addprefix Documentation for commit ,$(COMMIT_ID))

.PHONY: format
format:
	ruff format .

.PHONY: lint
lint:
	ruff check --preview .
	mypy .
	pylint src/

.PHONY: pre-commit
pre-commit:
	pre-commit install
	pre-commit run --all-files

.PHONY: docs
docs:
	splendor --version  # ensure splendor is installed.
	sphinx-apidoc --output-dir $(DOCS_SOURCE_DIR) $(SOURCE_DIR) --force
	sphinx-build $(DOCS_SOURCE_DIR) $(DOCS_BUILD_DIR)

.PHONY: publish-docs
publish-docs: docs
	pre-commit uninstall
	cd $(DOCS_BUILD_DIR) && git add -A . && git commit -sm "$(COMMIT_MSG)." && git push origin gh-pages
	pre-commit install

.PHONY: clean
clean:
	make -C docs/ clean

.PHONY: test
test:
	python -m pytest tests/

# The parity gate: mandatory before any web deployment after touching the
# engine's mask logic, features.py, or the browser extraction layer.
.PHONY: parity
parity:
	python -m pytest tests/test_feature_parity.py tests/test_browser_adapter.py -v

.PHONY: train-dqn
train-dqn:
	dqn --help && echo "usage: dqn -o random --test-opponent minimax [see plan/phase-1]"

.PHONY: play-web
play-web:
	play-web --help && echo "usage: play-web --checkpoint <path> --games 50 [see plan/phase-3]"
