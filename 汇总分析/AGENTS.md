# Repository Guidelines

## Project Structure & Module Organization
- Notebooks: `analysis.ipynb` is the primary analysis entry point.
- Data: CSVs in the repo root (e.g., `STK_LISTEDCOINFOANL.csv`, `month_code.csv`, `year_code.csv`). Prefer adding new datasets under `data/`.
- Code (optional): Put reusable helpers in `src/` (e.g., `src/utils.py`).
- Outputs: Save derived tables/figures to `output/` and keep it git-ignored.

## Build, Test, and Development Commands
- Create venv (Windows): `python -m venv .venv && .\.venv\Scripts\activate`
  - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
- Install basics: `pip install -U pandas jupyter matplotlib seaborn pytest black isort`
- Launch notebook: `jupyter lab` (or `jupyter notebook`) and open `analysis.ipynb`.
- Format/lint: `black src tests && isort src tests` (run before committing).
- Run tests (if present): `pytest -q`.

## Coding Style & Naming Conventions
- Python 3.9+; 4-space indentation; format with Black (line length 88) and import sort with isort.
- Names: `snake_case` for modules/functions/variables, `PascalCase` for classes. Example: `src/data_joiner.py` with `def build_sector_map(...)`.
- Notebooks: name descriptively, e.g., `20250914_listed_companies.ipynb`.
- Use type hints in `.py` files and small, pure functions for data transforms; keep heavy logic out of notebooks when possible.

## Testing Guidelines
- Use `pytest`; place tests in `tests/` named `test_*.py` (e.g., `tests/test_utils.py`).
- Add tests for any new function in `src/`; include minimal sample CSVs under `tests/fixtures/`.
- Focus on deterministic transforms and joins; prefer property-based checks for schema/row counts where helpful.

## Commit & Pull Request Guidelines
- Follow Conventional Commits: `feat|fix|docs|refactor|test|chore: summary`.
  - Example: `feat: add sector mapping and monthly join logic`
- PRs should include: concise description, linked issues, and sample outputs (PNG/table) from `output/` when relevant. Note any data/source assumptions.

## Security & Configuration Tips
- Do not commit sensitive or proprietary data. Keep large intermediates in `output/` and add to `.gitignore`; consider Git LFS for large, shareable assets.
- CSV encoding: prefer `utf-8`. Example: `pd.read_csv('STK_LISTEDCOINFOANL.csv', encoding='utf-8')`.
- Use relative paths with `pathlib.Path`, e.g., `Path('month_code.csv')`, to avoid OS-specific breakage.
- Before committing notebooks, clear outputs to reduce diffs: `jupyter nbconvert --clear-output --inplace analysis.ipynb`.

