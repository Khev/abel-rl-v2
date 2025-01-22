import os
import numpy as np

eqns = [
    "a*x",
    "x + b",
    "a*x + b",
    "a/x + b",
    "c*(a*x + b) + d",
    "sqrt(a*x+b) - c",
    "(a*x**2+b)**2 + c",
    "d/(a*x + b) + c",
    "e*(a*x + b) + (c*x + d)",
    "(a*x + b)/(c*x + d) + e"
]

train_letters = ['a','b','c','d','e']
test_letters  = ['f','g','h','i','j']

def find_placeholders(eqn_str):
    """Returns a sorted list of placeholders in eqn_str among {a,b,c,d,e}."""
    return [ch for ch in ['a','b','c','d','e'] if ch in eqn_str]

def cyc_expansions(eqn_str, letter_list, n_exp=5):
    """
    Generate n_exp expansions of eqn_str by cycling placeholders
    in eqn_str among letter_list.
    """
    placeholders = find_placeholders(eqn_str)
    if not placeholders:
        return [eqn_str]*n_exp  # No placeholders => just return repeated

    expansions = []
    for i in range(n_exp):
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

def generate_deep_data(eqns, max_level=10, outdir="equation_templates/deep"):
    """
    For each level i:
    - Train equations are the same as the shallow case.
    - Test equations come from level i+1 (i.e., unknown equation types).
    - Save them to .txt files.
    """
    os.makedirs(outdir, exist_ok=True)
    
    max_level = min(max_level, len(eqns))

    for lvl in range(max_level):
        eqn_set_train = eqns[:lvl+1]   # Train data as usual
        eqn_set_test  = eqns[:lvl+2] if lvl + 1 < max_level else eqns[:lvl+1]  # Test includes next level if possible

        train_all = []
        test_all = []
        for eqn_template in eqn_set_train:
            train_all.extend(cyc_expansions(eqn_template, train_letters, n_exp=5))
        
        for eqn_template in eqn_set_test:
            test_all.extend(cyc_expansions(eqn_template, test_letters, n_exp=5))
        
        level_dir = os.path.join(outdir, f"level{lvl}")
        os.makedirs(level_dir, exist_ok=True)

        save_list_to_txt(os.path.join(level_dir, "train_eqns.txt"), train_all)
        save_list_to_txt(os.path.join(level_dir, "test_eqns.txt"),  test_all)

        print(f"Level {lvl}: saved {len(train_all)} train eqns, {len(test_all)} test eqns in {level_dir}")

# Example usage
if __name__ == "__main__":
    generate_deep_data(eqns, max_level=10, outdir="equation_templates/deep")

