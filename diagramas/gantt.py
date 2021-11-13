import plotly.express as px
import pandas as pd

dataframe = pd.read_csv('gantt.csv', index_col=False)

fig = px.timeline(dataframe, x_start='Fecha Inicio', x_end='Fecha Fin', y='Proceso')
fig.update_yaxes(autorange='reversed')
fig.write_image('plantilla/img/gant.png')
