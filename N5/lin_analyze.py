import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load results
methods = ['iNN', 'NN', 'ODE', 'RANDOM', 'ENTROPY']
results = {}

for method in methods:
    try:
        mocu_data = np.loadtxt(f'results/{method}_MOCU.txt')
        if mocu_data.ndim == 1:
            mocu_data = mocu_data.reshape(1, -1)
        results[method] = mocu_data
        print(f"✓ Loaded {method}: {mocu_data.shape[0]} simulations")
    except:
        print(f"✗ Failed to load {method}")

# Calculate statistics
summary = []
for method, data in results.items():
    initial = np.mean(data[:, 0])
    final = np.mean(data[:, -1])
    improvement = (initial - final) / initial * 100
    
    summary.append({
        'Method': method,
        'Initial_MOCU': f"{initial:.4f}",
        'Final_MOCU': f"{final:.4f}",
        'Improvement': f"{improvement:.1f}%",
        'Simulations': data.shape[0]
    })

# Display results
df = pd.DataFrame(summary)
print("\n=== FINAL COMPARISON RESULTS ===")
print(df.to_string(index=False))

# Plot comparison
plt.figure(figsize=(12, 8))

# Plot 1: Final MOCU comparison
plt.subplot(2, 2, 1)
final_values = [float(row['Final_MOCU']) for row in summary]
methods_names = [row['Method'] for row in summary]
bars = plt.bar(methods_names, final_values)
plt.title('Final MOCU Values (Lower = Better)')
plt.ylabel('MOCU')
plt.xticks(rotation=45)

# Add values on bars
for bar, val in zip(bars, final_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom')

# Plot 2: Improvement percentages
plt.subplot(2, 2, 2)
improvements = [float(row['Improvement'].replace('%', '')) for row in summary]
bars = plt.bar(methods_names, improvements)
plt.title('MOCU Improvement Percentages')
plt.ylabel('Improvement (%)')
plt.xticks(rotation=45)

# Add values on bars
for bar, val in zip(bars, improvements):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom')

# Plot 3: Convergence curves
plt.subplot(2, 1, 2)
for method, data in results.items():
    mean_curve = np.mean(data, axis=0)
    experiments = range(len(mean_curve))
    plt.plot(experiments, mean_curve, 'o-', label=method, linewidth=2)

plt.xlabel('Experiment Number')
plt.ylabel('MOCU Value')
plt.title('MOCU Convergence Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/final_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as: results/final_comparison.png")