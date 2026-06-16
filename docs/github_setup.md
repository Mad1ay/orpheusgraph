# orpheusgraph — GitHub & Repository Setup

## Repository Structure

```
github.com/Mad1ay/orpheusgraph/
├── .github/
│   └── workflows/
│       ├── ci.yml          # Lint + test + benchmark on PR
│       └── release.yml     # Build wheels + publish to PyPI/crates.io
├── orpheusgraph/
│   ├── Cargo.toml
│   ├── pyproject.toml      # maturin config
│   ├── src/
│   │   ├── lib.rs
│   │   ├── graph.rs
│   │   ├── builder.rs
│   │   ├── overlay.rs
│   │   ├── scoring.rs
│   │   ├── traversal.rs
│   │   ├── serialization.rs
│   │   └── types.rs
│   ├── python/
│   │   └── orpheusgraph.pyi
│   ├── tests/
│   ├── benches/
│   └── examples/
├── docs/
│   ├── spec.md
│   ├── diagrams.md
│   └── sprints/
├── LICENSE                  # Proprietary
├── README.md
├── CONTRIBUTING.md
└── CHANGELOG.md
```

## GitHub Actions: CI

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test-rust:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all-features
      - run: cargo clippy -- -D warnings
      - run: cargo fmt --check

  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install maturin pytest
      - run: maturin develop
      - run: pytest tests/ -v

  benchmark:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo bench --bench bench_traversal -- --output-format bencher
```

## GitHub Actions: Release

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ["v*"]

jobs:
  build-wheels:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.10", "3.11", "3.12"]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install maturin
      - run: maturin build --release
      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.python }}
          path: target/wheels/*.whl

  publish-ghcr:
    needs: build-wheels
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - name: Upload wheels to GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: "**/*.whl"
```

## Initial Setup Commands

```bash
# 1. Create private repo
gh repo create Mad1ay/orpheusgraph --private --description "Rust knowledge graph traversal engine with Python bindings"

# 2. Init Rust crate
cargo init orpheusgraph --lib
cd orpheusgraph

# 3. Init maturin (Python bindings)
maturin init --bindings pyo3

# 4. First commit
git add .
git commit -m "feat: initial crate structure"
git push origin main
```

> ⚠️ **No public publishing.** Wheels are attached to GitHub Releases (private repo).
> Install from release: `pip install https://github.com/Mad1ay/orpheusgraph/releases/download/v0.1.0/orpheusgraph-0.1.0-cp311-*.whl`
> Or from source: `pip install git+ssh://git@github.com/Mad1ay/orpheusgraph.git`
```

## Branching Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable releases, tagged with `v0.x.x` |
| `dev` | Integration branch for PRs |
| `sprint/N` | Sprint branches (e.g. `sprint/1-core-types`) |

## Versioning

- Follow SemVer: `0.1.0` → `0.2.0` → `1.0.0`
- Pre-1.0: breaking changes allowed on minor bumps
- Each sprint = one minor version bump
