import os
import numpy as np

eqns = [
    "a*x",
    "x + b",
    "a*x + b",
    "a/x + b",
    "c*(a*x + b) + d",
    "d/(a*x + b) + c",
    "e*(a*x + b) + (c*x + d)",
    "(a*x + b)/(c*x + d) + e"
]

train_letters = ['a','b','c','d','e']
test_letters  = ['f','g','h','i','j']

def find_placeholders(eqn_str):
    """
    Returns a sorted list of placeholders in eqn_str among {a,b,c,d,e}.
    """
    return [ch for ch in ['a','b','c','d','e'] if ch in eqn_str]

def cyc_expansions(eqn_str, letter_list, n_eqns=2):
    """
    Generate n_eqns expansions of eqn_str by cycling placeholders
    in eqn_str among letter_list.
    """
    placeholders = find_placeholders(eqn_str)
    if not placeholders:
        return [eqn_str] * n_eqns  # No placeholders => just return eqn_str repeated

    expansions = []
    for i in range(n_eqns):
        eqn_modified = eqn_str
        for j, ph in enumerate(placeholders):
            letter_idx = (i + j) % len(letter_list)
            replacement = letter_list[letter_idx]
            eqn_modified = eqn_modified.replace(ph, replacement)
        expansions.append(eqn_modified)
    return expansions

def save_list_to_txt(filename, eqn_list):
    """Saves a list of equations into a text file, one per line."""
    with open(filename, "w") as f:
        for eqn in eqn_list:
            f.write(eqn + "\n")

def generate_lexical_generalization(eqns, n_eqns=5, outdir="equation_templates/lexical"):
    """
    For each level from 0..7 (fixed equation type), generate train/test sets
    with different constants.
    """
    os.makedirs(outdir, exist_ok=True)

    for lvl, eqn_template in enumerate(eqns[:8]):  # Levels 0 to 7
        train_eqns = cyc_expansions(eqn_template, train_letters, n_eqns=n_eqns)
        test_eqns = cyc_expansions(eqn_template, test_letters, n_eqns=n_eqns)

        level_dir = os.path.join(outdir, f"level{lvl}")
        os.makedirs(level_dir, exist_ok=True)

        save_list_to_txt(os.path.join(level_dir, "train_eqns.txt"), train_eqns)
        save_list_to_txt(os.path.join(level_dir, "test_eqns.txt"), test_eqns)

        print(f"Level {lvl}: {eqn_template} -> saved {len(train_eqns)} train, {len(test_eqns)} test in {level_dir}")

# Example usage
if __name__ == "__main__":
    generate_lexical_generalization(eqns, n_eqns=5, outdir="equation_templates/lexical")
