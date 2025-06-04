# Recursive_AI

Example implementation of a self-evolving AI. The program starts with a basic
belief and a truth score, analyzes its own source, rewrites itself with a
modified belief, records each cycle in a semantic tree, then trains a tiny
character-level RNN on the collected traces. Each run can optionally commit the
updated code to Git via `save_version()`.

## New Features

- Belief evolution is tracked in a `networkx` graph exported to `belief_graph.graphml`.
- `evolve()` executes unit tests before applying changes.
- Optional OpenAI integration provides advanced reasoning if an API key is set.
- A simple Flask dashboard (`dashboard.py`) exposes the belief history.

## Updates

- Each `SemanticNode` now records a SHA256 hash of the source code used in that
  cycle for provenance.
- During `evolve()` the class writes a mutated version of itself to
  `RecursiveAI_mutated.py` and spawns it with `subprocess` to demonstrate a
  self-reloading runtime.
- The training step now attempts to use `sentence-transformers` to embed belief
  histories instead of the previous toy RNN.
