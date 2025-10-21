# Repository Guidelines

## Project Structure & Module Organization
The repository centers on regression analysis assets. Raw disclosure datasets (`CG_*.csv`, `FS_*.csv`, `treat.csv`) live in the repository root; keep derived tables in `data/processed/` to avoid mixing sources. Place reusable Python modules under `src/` using snake_case filenames, and put exploratory notebooks in `notebooks/`. Narrative references, such as `background.md` and supporting PDFs, should stay in `docs/` for quick lookup.

## Build, Test, and Development Commands
Use a dedicated virtual environment before touching the data:
- `python -m venv .venv` creates an isolated interpreter.
- `.venv\Scripts\Activate.ps1` enables the environment on Windows PowerShell.
- `python -m pip install -r requirements.txt` installs analysis dependencies; update this file whenever you add a package.
- `python -m jupyter lab` launches notebook workspaces that point at the datasets above.

## Coding Style & Naming Conventions
Stick to PEP 8 with 4-space indentation. Name modules, functions, and variables with descriptive snake_case; reserve PascalCase for pandas DataFrame classes only. Column names imported from source files should remain unchanged; add comments mapping them to translations instead of renaming. Run `python -m black src notebooks` before committing, and pair it with `python -m isort src notebooks` to keep imports tidy.

## Testing Guidelines
Favor pytest for regression logic. Store unit tests alongside code in `tests/` mirroring the module path, and name files `test_<feature>.py`. Aim for >=80% coverage on newly introduced modules (`coverage run -m pytest` then `coverage html`). For data-driven tests, keep lightweight fixtures in `tests/fixtures/` so CI stays fast.

## Commit & Pull Request Guidelines
Write commits in the imperative mood (`Add panel merge helper`). Keep data drops isolated from code changes and reference the originating source in the message body. Pull requests should outline the motivation, summarize methodology, list datasets touched, and include before/after metrics or screenshots for notebook visualizations. Tag reviewers responsible for data governance whenever raw inputs change.

## Data Handling & Security
Treat all CSVs as confidential; do not upload raw files to third-party services or public issues. Scrub notebook outputs before pushing, and add `.gitattributes` rules if you need Git LFS. Back up processed artifacts to the shared drive instead of expanding the repository size.