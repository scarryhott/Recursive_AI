# Recursive_AI

Example implementation of a self-evolving AI. The program starts with a basic
belief and a truth score, analyzes its own source, rewrites itself with a
modified belief, records each cycle in a semantic tree, then trains a tiny
character-level RNN on the collected traces. Each run can optionally commit the
updated code to Git via `save_version()`.

## Updates

- Each `SemanticNode` now records a SHA256 hash of the source code used in that
  cycle for provenance.
- During `evolve()` the class writes a mutated version of itself to
  `RecursiveAI_mutated.py` and spawns it with `subprocess` to demonstrate a
  self-reloading runtime.
- The training step now attempts to use `sentence-transformers` to embed belief
  histories instead of the previous toy RNN.
