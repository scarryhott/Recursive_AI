# Recursive_AI

Example implementation of a self-evolving AI. The program starts with a basic
belief and a truth score, analyzes its own source, rewrites itself with a
modified belief, records each cycle in a semantic tree, then trains a tiny
character-level RNN on the collected traces. Each run can optionally commit the
updated code to Git via `save_version()`.
