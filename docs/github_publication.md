# GitHub Publication Guide

This repository should be published as a focused Python/ML project. Local
datasets, weights, generated outputs and the separate application prototype are
not required to reproduce the notebook/package structure.

## Include

| Path | Purpose |
| --- | --- |
| `src/` | Reusable Python package |
| `notebooks/` | Final experiment notebooks |
| `configs/` | Shared and experiment YAML configs |
| `tests/` | Unit and integration tests |
| `scripts/validate_notebooks.py` | Local notebook smoke test |
| `docs/` | Architecture, data and usage documentation |
| `.github/workflows/tests.yml` | Python CI |
| `README.md` | Project entry point |
| `.gitignore` | Local artifact exclusions |
| `pyproject.toml` | Package metadata and dependencies |
| `requirements-datasphere.txt` | Notebook/training environment |
| `requirements-preprocessing.txt` | Optional preprocessing environment |

## Do Not Include In This ML Push

- `data/`, `datasets/`, `checkpoints/`, `outputs/`, `artifacts/`;
- `*.npz`, `*.npy`, `*.dat`, `*.raw`, `*.sgy`, `*.segy`;
- `*.pth`, `*.pt`, `*.ckpt`, generated `*.html`;
- `.venv/`, nested `venv/`, `node_modules/`, `.next/`;
- `.agents/`, `.codex/`, IDE state and local environment files;
- `apps/`, `packages/`, `infra/` and root JavaScript/Docker files, because
  they belong to a separate application workspace;
- root helper files such as `create_npz.py`, `upload_script.py`,
  `replace_token.js`, `dummy.segy` and `valid.npz`.

## Stage The Intended Files

Do not use `git add .` while unrelated workspace files are present. From the
repository root use the explicit allowlist:

```bash
git add -u
git add \
  .gitignore \
  README.md \
  pyproject.toml \
  requirements-datasphere.txt \
  requirements-preprocessing.txt \
  configs \
  docs \
  notebooks \
  scripts/validate_notebooks.py \
  src \
  tests \
  .github/workflows/tests.yml
```

`git add -u` records removal of the replaced root-level `.ipynb` files that
are already tracked. The final notebook set lives only in `notebooks/`.

Review the staged snapshot:

```bash
git status --short
git diff --cached --stat
git diff --cached --check
```

Before the first public release, also decide on a license and replace any
placeholder repository URL used in citation metadata.
