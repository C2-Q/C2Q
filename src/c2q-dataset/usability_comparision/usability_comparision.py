import os

# Path to the folder with problem example scripts
folder_path = '../usability_comparision/problem_examples_usability'

# Dictionary to store results
loc_counts = {}

# Loop over each .py file
for filename in os.listdir(folder_path):
    if filename.endswith('.py'):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # Filter out comments and blank lines
            effective_lines = [
                line for line in lines
                if line.strip() and not line.strip().startswith('#')
            ]
            loc_counts[filename] = len(effective_lines)

# Print individual results
total_lines = 0
for filename, loc in sorted(loc_counts.items()):
    print(f"{filename:20} → {loc} lines")
    total_lines += loc

# Calculate and print average
if loc_counts:
    average = total_lines / len(loc_counts)
    print(f"\nAverage effective LoC: {average:.2f}")
else:
    print("\nNo Python files found.")