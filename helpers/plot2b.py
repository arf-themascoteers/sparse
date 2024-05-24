import pandas as pd
import plotly.express as px

df = pd.read_csv("combo_2b_filtered.csv")

fig = px.scatter(df, x='band1', y='band2',
                    color='score',
                    color_continuous_scale='ylgnbu',
                    labels={"band1": "Band Index 1", "band2": "Band Index 2"}
                    )

fig.update_layout({
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'title_x': 0.5
})

fig.update_traces(marker=dict(size=5))

fig.update_layout(
    xaxis=dict(range=[0, 66]),
    yaxis=dict(range=[0, 66])
)

fig.update_layout(
    font=dict(size=22),
)

fig.update_layout(
    coloraxis_colorbar=dict(
        title_text=''  # Ensures no vertical title is displayed
    ),
    width=700,
    height=700,
)

fig.add_annotation(
    x=1.05, y=1.01,  # Position of the annotation near the colorbar
    text="$R^2$",  # Your custom title
    showarrow=False,
    xref="paper", yref="paper",
    xanchor="left", yanchor="middle",
    textangle=0,
    font=dict(  # Setting font size and family
        size=25,
        family="Arial"
    )
)

fig.write_image("fig1.png", scale=5)