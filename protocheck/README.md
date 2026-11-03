# protocheck (vendored copy, frozen at v0.1.0)

**The canonical home of this tool is
[github.com/UlisesMoyaSanchez/protocheck](https://github.com/UlisesMoyaSanchez/protocheck).**
Install it, report issues and read the documentation there:

```bash
pip install "git+https://github.com/UlisesMoyaSanchez/protocheck.git@v0.1.0"
```

This folder is a **frozen copy of release v0.1.0** (`src/protocheck/` at tag `v0.1.0`), kept here
only so that the paper's analysis scripts import exactly the code that produced the reported
numbers, without an extra install step:

- `analysis/validate_protocheck.py`, `analysis/validate_protocheck_ogb.py`
- `analysis/protocheck_mgcnss.py`
- `analysis/compare_matched_samplers.py`, `analysis/test_fallback_policy.py`
- `tests/test_protocheck.py`

**Do not edit it here.** Changes go to the standalone repository first; to update this copy,
copy `src/protocheck/` from the new tag over this folder (keeping this README and
`requirements.txt`) and rerun `pytest tests/test_protocheck.py` and
`analysis/validate_protocheck.py`.
