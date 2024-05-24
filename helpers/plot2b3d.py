import pandas as pd
import plotly.express as px

df = pd.read_csv("b22.csv")

fig = px.scatter_3d(df, x='band1', y='band2', z='score',
                    color='score',
                    color_continuous_scale='Viridis',
                    title='3D Scatter Plot',
                    opacity=0.5
                    )
fig.show()