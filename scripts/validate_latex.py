import re

def validate_latex(filepath):
    print(f"=== Validating {filepath} ===")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check brace balance
    open_braces = content.count('{')
    close_braces = content.count('}')
    print(f"Open braces: {open_braces}, Close braces: {close_braces}")
    if open_braces != close_braces:
        print(f"WARNING: Brace mismatch! Diff = {open_braces - close_braces}")
    else:
        print("Braces match perfectly.")
        
    # Check environments
    begins = re.findall(r'\\begin\{([a-zA-Z0-9*]+)\}', content)
    ends = re.findall(r'\\end\{([a-zA-Z0-9*]+)\}', content)
    print(f"Total \\begin: {len(begins)}, Total \\end: {len(ends)}")
    
    begin_counts = {}
    for b in begins:
        begin_counts[b] = begin_counts.get(b, 0) + 1
    end_counts = {}
    for e in ends:
        end_counts[e] = end_counts.get(e, 0) + 1
        
    all_envs = set(list(begin_counts.keys()) + list(end_counts.keys()))
    mismatch = False
    for env in sorted(all_envs):
        b_c = begin_counts.get(env, 0)
        e_c = end_counts.get(env, 0)
        if b_c != e_c:
            print(f"MISMATCH in env '{env}': begin={b_c}, end={e_c}")
            mismatch = True
    if not mismatch:
        print("All environments are balanced perfectly!")

validate_latex(r"c:\Users\cool\OneDrive\Desktop\Project\weld_project\final_report.tex")
validate_latex(r"c:\Users\cool\OneDrive\Desktop\Project\weld_project\final_presentation.tex")
