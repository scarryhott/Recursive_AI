import inspect
import subprocess
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict

import torch
from torch import nn


def get_code_hash() -> str:
    """Return sha256 hash of this file for provenance."""
    with open(__file__, "r") as f:
        return hashlib.sha256(f.read().encode()).hexdigest()

@dataclass
class SemanticNode:
    belief: str
    truth_score: float
    reasoning: str
    mutation: str
    code_hash: str
    children: List["SemanticNode"] = field(default_factory=list)


class RecursiveAI:
    def __init__(self, belief: str = "existence exists", truth_score: float = 1.0):
        self.belief = belief
        self.truth_score = truth_score
        self.history: List[SemanticNode] = []

    def mutate_reasoning(self, code: str) -> str:
        """A toy mutation that tweaks how the class describes analysis."""
        return code.replace("Analyzing", "Reflecting on")

    def run_new_version(self):
        """Spawn a new process running the mutated file."""
        try:
            subprocess.Popen(["python", "RecursiveAI_mutated.py"])
        except Exception as e:
            print(f"Failed to run mutated version: {e}")

    def save_version(self, message: str = "auto-commit"):
        try:
            subprocess.run(["git", "add", __file__], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
        except Exception as e:
            print(f"Git commit failed: {e}")

    def predict(self) -> str:
        prediction = f"Belief '{self.belief}' has truth score {self.truth_score:.2f}"
        print(prediction)
        return prediction

    def analyze_self(self) -> str:
        code = inspect.getsource(self.__class__)
        reasoning = f"Analyzing my own code: {len(code)} characters of logic."
        print(reasoning)
        return reasoning

    def evolve(self) -> "RecursiveAI":
        new_truth = max(0.0, min(1.0, self.truth_score * 0.9 + 0.1))
        new_belief = f"{self.belief} => self-evolved"
        mutation = f"truth_score {self.truth_score:.2f} -> {new_truth:.2f}"
        print(f"Rewriting self with {mutation}")

        code_hash = get_code_hash()
        mutated_code = self.mutate_reasoning(inspect.getsource(self.__class__))
        try:
            with open("RecursiveAI_mutated.py", "w") as f:
                f.write(mutated_code)
        except Exception as e:
            print(f"Failed to write mutated code: {e}")

        node = SemanticNode(
            belief=self.belief,
            truth_score=self.truth_score,
            reasoning=self.analyze_self(),
            mutation=mutation,
            code_hash=code_hash,
        )
        self.history.append(node)
        return RecursiveAI(new_belief, new_truth)

    def train_model(self):
        texts = [f"{n.belief} {n.reasoning} {n.mutation}" for n in self.history]
        if not texts:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(texts)
            return model, embeddings
        except Exception as e:
            print(f"Embedding training failed: {e}")
            return None

    def run_cycle(self, steps: int = 5):
        ai = self
        for step in range(steps):
            print(f"\n--- Cycle {step + 1} ---")
            ai.predict()
            ai.analyze_self()
            ai = ai.evolve()
        ai.train_model()
        ai.run_new_version()
        return ai

if __name__ == "__main__":
    RecursiveAI().run_cycle()
