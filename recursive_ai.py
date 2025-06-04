import inspect
import subprocess
import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import networkx as nx
import openai



def get_code_hash() -> str:
    """Return sha256 hash of this file for provenance."""
    with open(__file__, "r") as f:
        return hashlib.sha256(f.read().encode()).hexdigest()

@dataclass
class SemanticNode:
    id: int
    belief: str
    truth_score: float
    reasoning: str
    mutation: str
    code_hash: str
    children: List["SemanticNode"] = field(default_factory=list)


class RecursiveAI:
    def __init__(self, belief: str = "existence exists", truth_score: float = 1.0, init_graph: bool = True):
        self.belief = belief
        self.truth_score = truth_score
        self.history: List[SemanticNode] = []
        self.graph = nx.DiGraph()
        self.node_counter = 0
        if init_graph:
            root = SemanticNode(
                id=self.node_counter,
                belief=self.belief,
                truth_score=self.truth_score,
                reasoning="initial belief",
                mutation="init",
                code_hash=get_code_hash(),
            )
            self.history.append(root)
            self.graph.add_node(root.id, **root.__dict__)
            self.last_node_id = root.id
            self.node_counter += 1

    def mutate_reasoning(self, code: str) -> str:
        """A toy mutation that tweaks how the class describes analysis."""
        return code.replace("Analyzing", "Reflecting on")

    def run_unit_tests(self) -> bool:
        """Run unit tests to ensure integrity before evolution."""
        if os.getenv("SKIP_SELF_TESTS"):
            return True
        try:
            result = subprocess.run(
                ["python", "-m", "unittest", "discover", "-s", "tests"],
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Tests failed:\n{e.stdout}\n{e.stderr}")
            return False

    def export_graph(self, path: str = "belief_graph.graphml"):
        """Save the belief evolution graph to a file."""
        try:
            nx.write_graphml(self.graph, path)
        except Exception as e:
            print(f"Failed to save graph: {e}")

    def run_new_version(self):
        """Spawn a new process running the mutated file."""
        try:
            subprocess.Popen(["python", "RecursiveAI_mutated.py"])
        except Exception as e:
            print(f"Failed to run mutated version: {e}")

    def document_evolution(self, node: SemanticNode, test_passed: bool):
        """Append evolution details to a report file."""
        report = (
            f"Belief: {node.belief}\n"
            f"Truth score: {node.truth_score}\n"
            f"Reasoning: {node.reasoning}\n"
            f"Mutation: {node.mutation}\n"
            f"Code hash: {node.code_hash}\n"
            f"Tests passed: {test_passed}\n"
            "---\n"
        )
        try:
            with open("evolution_report.txt", "a") as f:
                f.write(report)
        except Exception as e:
            print(f"Failed to write report: {e}")

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

    def advanced_reasoning(self, prompt: str) -> Optional[str]:
        """Use OpenAI's API for advanced reasoning if API key available."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Advanced reasoning failed: {e}")
            return None

    def analyze_self(self) -> str:
        code = inspect.getsource(self.__class__)
        reasoning = f"Analyzing my own code: {len(code)} characters of logic."
        extra = self.advanced_reasoning("Summarize the purpose of this code.")
        if extra:
            reasoning += " " + extra
        print(reasoning)
        return reasoning

    def evolve(self) -> "RecursiveAI":
        if not self.run_unit_tests():
            print("Evolution aborted due to failing tests.")
            return self

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
            id=self.node_counter,
            belief=self.belief,
            truth_score=self.truth_score,
            reasoning=self.analyze_self(),
            mutation=mutation,
            code_hash=code_hash,
        )
        self.history.append(node)
        self.graph.add_node(node.id, **node.__dict__)
        self.graph.add_edge(self.last_node_id, node.id)
        self.document_evolution(node, True)
        self.last_node_id = node.id
        self.node_counter += 1

        self.belief = new_belief
        self.truth_score = new_truth
        return self

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
        for step in range(steps):
            print(f"\n--- Cycle {step + 1} ---")
            self.predict()
            self.analyze_self()
            self = self.evolve()
        self.train_model()
        self.export_graph()
        self.run_new_version()
        return self

if __name__ == "__main__":
    RecursiveAI().run_cycle()
