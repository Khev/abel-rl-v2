import os
import numpy as np

eqns = [
    "a*x",
    "x + b",
    "a*x + b",
    "a/x + b",
    "c*(a*x + b) + d",
    # "sqrt(a*x+b) - c",
    # "(a*x**2+b)**2 + c",
    "d/(a*x + b) + c",
    "e*(a*x + b) + (c*x + d)",
    "(a*x + b)/(c*x + d) + e"
]

train_letters = ['a','b','c','d','e']
test_letters  = ['f','g','h','i','j']

def find_placeholders(eqn_str):
    """
    Returns a sorted list of placeholders in eqn_str among {a,b,c,d,e}.
    e.g. eqn_str = 'a*x + b' => placeholders = ['a','b']
    """
    placeholders = []
    for ch in ['a','b','c','d','e']:  # The placeholders we consider for eqns
        if ch in eqn_str:
            placeholders.append(ch)
    return placeholders

def cyc_expansions(eqn_str, letter_list, n_eqns=2):
    """
    Generate n_eqns expansions of eqn_str by cycling placeholders
    in eqn_str among letter_list.
    - eqn_str might have placeholders from 'a','b','c','d','e'
    - letter_list is typically 5 letters (train or test).
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
    """
    Saves a list of equations into a text file, one per line.
    """
    with open(filename, "w") as f:
        for eqn in eqn_list:
            f.write(eqn + "\n")

def generate_data(eqns, max_level=10, n_eqns=2, outdir="equation_templates/shallow"):
    """
    For each level from 0..(max_level-1),
    - eqn_set = eqns[:level+1]
    - produce n_eqns expansions for each eqn in eqn_set using train_letters => train_eqns
    - same expansions using test_letters => test_eqns
    - save them to .txt files
    """
    os.makedirs(outdir, exist_ok=True)

    max_level = min(max_level, len(eqns))

    for lvl in range(max_level):
        eqn_set = eqns[:lvl+1]  # e.g. lvl=0 => eqns[:1], lvl=1 => eqns[:2] etc.

        train_all = []
        test_all = []
        for eqn_template in eqn_set:
            train_versions = cyc_expansions(eqn_template, train_letters, n_eqns=n_eqns)
            test_versions = cyc_expansions(eqn_template, test_letters, n_eqns=n_eqns)

            train_all.extend(train_versions)
            test_all.extend(test_versions)

        level_dir = os.path.join(outdir, f"level{lvl}")
        os.makedirs(level_dir, exist_ok=True)

        save_list_to_txt(os.path.join(level_dir, "train_eqns.txt"), train_all)
        save_list_to_txt(os.path.join(level_dir, "test_eqns.txt"), test_all)

        print(f"Level {lvl}: saved {len(train_all)} train eqns, {len(test_all)} test eqns in {level_dir}")

# Example usage
if __name__ == "__main__":
    generate_data(eqns, max_level=10, n_eqns=2, outdir="equation_templates/shallow")
