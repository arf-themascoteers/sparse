import plotly.graph_objects as go
import numpy as np

# Define sparse data
x = np.linspace(0, 10, 30)  # Reduced number of x points
y = np.sin(x)  # Basic sine function

# Available markers in Plotly
markers = [
    "triangle-right", "triangle-right-open", "triangle-right-dot", "triangle-right-open-dot",
    "triangle-ne", "triangle-ne-open", "triangle-ne-dot", "triangle-ne-open-dot",
    "triangle-se", "triangle-se-open", "triangle-se-dot", "triangle-se-open-dot",
    "triangle-sw", "triangle-sw-open", "triangle-sw-dot", "triangle-sw-open-dot",
    "triangle-nw", "triangle-nw-open", "triangle-nw-dot", "triangle-nw-open-dot",
    "pentagon", "pentagon-open", "pentagon-dot", "pentagon-open-dot",
    "hexagon", "hexagon-open", "hexagon-dot", "hexagon-open-dot",
    "hexagon2", "hexagon2-open"
]

# Ensure we only take the first 30 markers
markers = markers[:30]

# Create figure
fig = go.Figure()

# Add lines with different markers
for i, marker in enumerate(markers):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y + i * 0.5,  # Increment y to separate the lines
            mode='lines+markers',
            name=marker,
            marker=dict(symbol=marker, size=10)
        )
    )

# Set up layout
fig.update_layout(
    title='Sparse Lines with Different Markers',
    xaxis_title='X',
    yaxis_title='Y',
    legend_title='Marker Type'
)

fig.show()
