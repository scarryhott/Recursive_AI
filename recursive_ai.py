import inspect
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict

import torch
from torch import nn

@dataclass
class SemanticNode:
    belief: str
    truth_score: float
    reasoning: str
    mutation: str
    children: List["SemanticNode"] = field(default_factory=list)


class RecursiveAI:
    def __init__(self, belief: str = "existence exists", truth_score: float = 1.0):
        self.belief = belief
        self.truth_score = truth_score
        self.history: List[SemanticNode] = []

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
        node = SemanticNode(
            belief=self.belief,
            truth_score=self.truth_score,
            reasoning=self.analyze_self(),
            mutation=mutation,
        )
        self.history.append(node)
        return RecursiveAI(new_belief, new_truth)

    def train_model(self):
        texts = [f"{n.belief} {n.reasoning} {n.mutation}" for n in self.history]
        if not texts:
            return None
        # Very small char-level RNN for demonstration
        vocab = sorted(set(" ".join(texts)))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = {i: ch for ch, i in stoi.items()}
        encoded = [torch.tensor([stoi[c] for c in t], dtype=torch.long) for t in texts]
        model = nn.RNN(len(vocab), 8, batch_first=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        for e in range(2):
            for seq in encoded:
                x = nn.functional.one_hot(seq[:-1], num_classes=len(vocab)).float().unsqueeze(0)
                y = seq[1:].unsqueeze(0)
                out, _ = model(x)
                loss = loss_fn(out.squeeze(0), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model, stoi, itos

    def run_cycle(self, steps: int = 5):
        ai = self
        for step in range(steps):
            print(f"\n--- Cycle {step + 1} ---")
            ai.predict()
            ai.analyze_self()
            ai = ai.evolve()
        ai.train_model()
        return ai

if __name__ == "__main__":
    RecursiveAI().run_cycle()
