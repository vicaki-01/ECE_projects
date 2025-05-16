print(torch.__version__)

import torch
import matplotlib.pyplot as plt
import networkx as nx

# Define the function
def f(x, y):
    return (x**2 + y)**3

# Create tensors with requires_grad=True
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Compute the function
result = f(x, y)

# Compute gradients
result.backward()

# Print gradients
print(f"∂f/∂x at x=2, y=3: {x.grad.item()}")
print(f"∂f/∂y at x=2, y=3: {y.grad.item()}")

# Manual verification
# f(x,y) = (x² + y)³
# ∂f/∂x = 3(x² + y)² * 2x = 6x(x² + y)²
# ∂f/∂y = 3(x² + y)²
manual_df_dx = 6 * 2 * (2**2 + 3)**2
manual_df_dy = 3 * (2**2 + 3)**2

print(f"Manual ∂f/∂x: {manual_df_dx}")
print(f"Manual ∂f/∂y: {manual_df_dy}")

# Create computation graph visualization
plt.figure(figsize=(8, 6))
G = nx.DiGraph()

# Add nodes with labels
G.add_node("x", label="x (input)")
G.add_node("y", label="y (input)")
G.add_node("x²", label="x²")
G.add_node("+", label="+")
G.add_node("()³", label="()³")

# Add edges
G.add_edge("x", "x²")
G.add_edge("x²", "+")
G.add_edge("y", "+")
G.add_edge("+", "()³")

# Draw the graph
pos = {
    "x": (0, 0),
    "y": (0, 1),
    "x²": (1, 0),
    "+": (2, 0.5),
    "()³": (3, 0.5)
}

labels = {n: G.nodes[n]['label'] for n in G.nodes()}
nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000, 
        node_color="skyblue", font_size=12, font_weight="bold",
        arrowsize=20)

plt.title("Computation Graph for f(x,y) = (x² + y)³", pad=20)
plt.savefig("computation_graph.png", bbox_inches='tight', dpi=300)
plt.close()

print("Computation graph saved as computation_graph.png")