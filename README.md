# Spot-prediction

Lightweight repo for stock prediction scripts and notebooks.


Getting started

- Install dependencies (pinned, reproducible):

```powershell
python -m pip install -r requirements.txt
```

- Development (use pip-tools):

1. Install pip-tools (only once):

```powershell
python -m pip install pip-tools
```

2. Install top-level dev deps during development:

```powershell
python -m pip install -r requirements-dev.in
```

3. To regenerate pinned files (run locally):

```powershell
pip-compile requirements.in
pip-compile requirements-dev.in -o requirements-dev.txt
```

- Run the main script:

```powershell
python main.py
```

Notes

- Data files are stored in the `Data/` folder and are ignored by git by default. If you want to track specific example datasets, move them to another folder or remove the entry from `.gitignore`.

Dependency management (recommended)

- Keep `requirements.in` and `requirements-dev.in` as top-level, editable manifests.
- Use `pip-compile` to produce `requirements.txt` (pinned lock) for CI and reproducible installs.
- CI installs from `requirements.txt` to ensure deterministic builds.

License

This project is provided under the MIT license. See `LICENSE`.
