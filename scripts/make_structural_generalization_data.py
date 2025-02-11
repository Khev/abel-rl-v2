import os

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

train_letters = ['a', 'b', 'c', 'd', 'e']
test_letters = ['f', 'g', 'h', 'i', 'j']

def find_placeholders(eqn_str):
    """Finds the unique variable placeholders in an equation."""
    return sorted(set(ch for ch in "abcde" if ch in eqn_str))

def unique_expansions(eqn_str, letter_list, n_eqns=1):
    """
    Generates `n_eqns` versions of `eqn_str`, ensuring each variable placeholder 
    is mapped to a unique letter from `letter_list` in each equation.
    """
    placeholders = find_placeholders(eqn_str)
    if len(placeholders) > len(letter_list):
        raise ValueError("Not enough letters to replace all placeholders uniquely.")

    expansions = []
    for i in range(n_eqns):
        shuffled_letters = letter_list[i % len(letter_list):] + letter_list[:i % len(letter_list)]
        replacement_map = {ph: rep for ph, rep in zip(placeholders, shuffled_letters)}
        
        eqn_modified = "".join(replacement_map.get(ch, ch) for ch in eqn_str)
        expansions.append(eqn_modified)

    return expansions

def save_list_to_txt(filename, eqn_list):
    """Saves a list of equations to a text file."""
    with open(filename, "w") as f:
        for eqn in eqn_list:
            f.write(eqn + "\n")

def generate_lexical_data(eqns, outdir="equation_templates/structural"):
    os.makedirs(outdir, exist_ok=True)

    for lvl in range(1, len(eqns) + 1):  # Iterate over levels, selecting eqns[:lvl]
        selected_eqns = eqns[:lvl+1]  # Use only the first `lvl` equations
        
        train_all = []
        test_all = []
        for eqn_template in selected_eqns:
            train_all.extend(unique_expansions(eqn_template, train_letters, n_eqns=2))
            test_all.extend(unique_expansions(eqn_template, test_letters, n_eqns=2))

        level_dir = os.path.join(outdir, f"level{lvl}")
        os.makedirs(level_dir, exist_ok=True)

        save_list_to_txt(os.path.join(level_dir, "train_eqns.txt"), train_all)
        save_list_to_txt(os.path.join(level_dir, "test_eqns.txt"), test_all)

        print(f"Level {lvl}: saved {len(train_all)} train eqns, {len(test_all)} test eqns in {level_dir}")

if __name__ == "__main__":
    generate_lexical_data(eqns)
