import os
import json
import matplotlib.pyplot as plt
import numpy as np

def save_results(filename, results):
        """Save results to a file."""
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

# Directory containing the JSON files
input_dir = "results"
# Output file
output_file = "combined_results.json"

combined = {}
# Iterate over all files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r") as f:
            data = json.load(f)
            combined.update(data)  # Merge the model results


# Save combined result
save_results(output_file, combined)

print(f"Combined JSON saved to {output_file}")

# Extract required data
data_points = []
for model_name, metrics in combined.items():
    param_num = metrics.get("param_num")
    perplexity = metrics.get("avg_normalized_perplexity")
    pass_at_1 = metrics.get("pass_at_k", {}).get("plus", {}).get("pass@1")
    if param_num is not None and perplexity is not None and pass_at_1 is not None:
        data_points.append((model_name, param_num, perplexity, pass_at_1 * 100))

# Sort by param_num
data_points.sort(key=lambda x: x[1])

# Unpack for plotting
model_names, param_nums, perplexities, pass_at_1s = zip(*data_points)

# Chart 1: Accuracy
plt.figure(figsize=(10, 5))
plt.bar(model_names, pass_at_1s, color='skyblue')
plt.title("Pass@1")
plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("model_accuracy.png")
plt.show()

# Chart 2: Generated Perplexity
plt.figure(figsize=(10, 5))
plt.bar(model_names, perplexities, color='lightgreen')
plt.title("Generated Perplexity")
plt.ylabel("Perplexity")
plt.xlabel("Model")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("model_perplexity.png")
plt.show()