# Contributing to classifai

Thank you for considering a contribution. This document explains how to get
involved, what we expect from contributors, and how to submit changes.

## Ways to contribute

- **Report a bug** — open an issue using the Bug Report template
- **Request a feature** — open an issue using the Feature Request template
- **Improve documentation** — fix typos, add examples, clarify explanations
- **Add a new backend** — extend classifai to support a new model or library
- **Improve examples** — add notebooks or improve existing ones
- **Write tests** — increase coverage or add edge cases

## Getting started

```bash
# 1. Fork the repo and clone your fork
git clone https://github.com/<your-username>/classifai.git
cd classifai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Create a branch
git checkout -b feature/my-improvement
```

## Development guidelines

**Code style**
- Format with `black` and sort imports with `isort` before committing
- Type hints are encouraged on public functions
- Docstrings on any new public function or class

**Tests**
- Every new backend or feature must include tests
- Run the test suite with `pytest` before submitting
- Aim for coverage on the happy path and at least one error path

**Commits**
- Use clear, present-tense commit messages: `Add zero-shot backend`, not `Added zero-shot backend`
- Keep commits focused — one logical change per commit

**Pull requests**
- Fill out the PR template completely
- Link to any related issue (e.g., `Closes #42`)
- Keep PRs focused; large refactors should be discussed in an issue first
- CI must pass before review

## Adding a new backend

Each backend lives in `classifai/backends/`. To add one:

1. Create `classifai/backends/my_backend.py` implementing the `BaseBackend` interface
2. Register it in `classifai/backends/__init__.py`
3. Add its optional dependencies to `pyproject.toml` under `[project.optional-dependencies]`
4. Add at least one test in `tests/backends/test_my_backend.py`
5. Document it in `README.md` under **Backends**

## Reporting issues

Before opening an issue:
- Search existing issues to avoid duplicates
- Include your Python version, OS, and package version
- Paste the full traceback, not just the last line

## Code of Conduct

Be respectful and constructive. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Harassment or discrimination of any kind will not be tolerated.

## Questions?

Open a [Discussion](https://github.com/JuanLara18/classifai/discussions) — issues
are for bugs and feature requests only.
