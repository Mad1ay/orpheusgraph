# Contributing to orpheusgraph

## License of contributions

`orpheusgraph` is dual-licensed: source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE.md) and available under separate
[commercial licenses](COMMERCIAL.md). By opening a pull request you agree to the
[Contributor License Agreement](CLA.md), which lets the maintainer include your
contribution in both the noncommercial and commercial editions. Please read it
before submitting.

## Prerequisites

- **Rust**: stable toolchain (`rustup default stable`)
- **Python**: 3.11+
- **maturin**: `pip install maturin`

## Development Setup

```bash
git clone https://github.com/Mad1ay/orpheusgraph.git
cd orpheusgraph

# Build and install locally
maturin develop

# Verify Python import
python -c "import orpheusgraph; print('OK')"
```

## Running Tests

```bash
# Rust unit tests (34 tests across 5 modules)
cargo test --all

# Clippy (must pass with zero warnings)
cargo clippy --all-targets -- -D warnings

# Benchmarks (50K node graph)
cargo bench

# Python smoke test (requires maturin develop first)
python -c "import orpheusgraph; print('OK')"
```

## Code Style

- **Rust**: standard `rustfmt`. Run `cargo fmt` before committing.
- **Clippy**: all warnings treated as errors in CI.
- **Python**: follows project-level ruff config.

## Architecture

```
src/
├── types.rs          # Data types (NodeData, EdgeData, DynamicContext, etc.)
├── builder.rs        # Graph construction + PageRank
├── graph.rs          # Immutable DiGraph wrapper
├── accessor.rs       # GraphAccessor trait (unified owned + archived access)
├── scoring.rs        # Scoring formula (multiplicative noise)
├── overlay.rs        # Virtual overlay + max_fan_out cutoff
├── traversal.rs      # Beam search, Dijkstra, subgraph extraction
├── serialization.rs  # rkyv zero-copy serialization
├── pybridge.rs       # PyO3 Python bindings + overlay cache
└── lib.rs            # Module entry point
```

## PR Process

1. Fork and create a feature branch
2. Ensure `cargo test --all` and `cargo clippy` pass
3. Update `CHANGELOG.md` if adding user-facing changes
4. Open PR — CI runs `rust-check` and `bench-gate` (10% regression threshold)
