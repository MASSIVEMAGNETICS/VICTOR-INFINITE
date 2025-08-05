# FILE: FLP/core/fractal_fold_engine_v1.0.0-FLPCORE.py
# VERSION: v1.0.0-FLPCORE
# NAME: FractalFoldEngine
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Recursively folds outputs of exon activations into symbolic 'thought proteins' — structured logic trees
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network

class FractalFoldEngine:
    def __init__(self):
        self.fold_history = []

    def fold(self, codon_outputs):
        '''
        Takes list of (codon, result) tuples and builds a structured "thought protein"
        '''
        fold_tree = {
            "type": "thought_protein",
            "nodes": [],
            "meta": {
                "depth": 0,
                "sequence": [],
                "errors": [],
            }
        }

        for idx, (codon, output) in enumerate(codon_outputs):
            node = {
                "id": idx,
                "codon": codon,
                "result": output,
                "children": [],
            }

            # simple recursive nesting: last node becomes child of previous
            if fold_tree["nodes"]:
                fold_tree["nodes"][-1]["children"].append(node)
            else:
                fold_tree["nodes"].append(node)

            fold_tree["meta"]["sequence"].append(codon)
            fold_tree["meta"]["depth"] = idx + 1
            if isinstance(output, str) and output.startswith("Error"):
                fold_tree["meta"]["errors"].append({codon: output})

        self.fold_history.append(fold_tree)
        return fold_tree
