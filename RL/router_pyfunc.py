"""MLflow pyfunc for the router policy, logged as "models from code" by tracking.register_policy.

Input: rows of the 8-d query context (CONTEXT_COLUMNS: three corpus features, the one-hot
question category, a bias). Output: the chosen engine and each engine's predicted reward, using
the same greedy rule as serving (corpus_router / run_public_benchmark.choose_frozen). Self-contained
(numpy, pandas) so a registered version loads without the app's code."""
import json

import mlflow
import numpy as np
import pandas as pd

CONTEXT_COLUMNS = ["cross_doc_entity_fraction", "entity_recurrence_depth", "num_docs_normalized",
                   "local", "global", "multi_hop", "conflict", "bias"]


class RouterModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["policy"], encoding="utf-8") as f:
            data = json.load(f)
        self.arms = data["arms"]
        self.theta = {a: np.linalg.solve(np.array(data["A"][a]), np.array(data["b"][a])) for a in self.arms}

    def predict(self, context, model_input, params=None):
        x = np.atleast_2d(np.asarray(model_input, dtype=float))
        rewards = {a: x @ self.theta[a] for a in self.arms}
        # Ties go to the earlier arm, as in serving.
        chosen = [max(self.arms, key=lambda a: (rewards[a][i], -self.arms.index(a))) for i in range(len(x))]
        return pd.DataFrame({"engine": chosen, **{f"reward_{a}": rewards[a] for a in self.arms}})


mlflow.models.set_model(RouterModel())
