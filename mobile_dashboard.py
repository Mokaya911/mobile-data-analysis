import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv(r"C:\Users\user\Desktop\data analysis project\datasets\datasetsCleaned_Mobile_Dataset.csv")

# Filter to stay within desired value ranges
df = df[
    (df['nched Price (USA)'] <= 5000) &
    (df['RAM'] <= 20) &
    (df['Battery Capacity'] <= 20000) &
    (df['Main Back Camera'] <= 250) &
    (df['Launched Year']<=2026)
]

# KMeans clustering
features = df[['RAM', 'Main Back Camera', 'Battery Capacity', 'nched Price (USA)']].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features)
cluster_series = pd.Series(clusters, index=features.index)
df['Customer Segment'] = df.index.map(cluster_series)
df['Customer Segment'] = df['Customer Segment'].fillna(-1).astype(int)

# Initialize Dash app
app = Dash(__name__)
app.title = "📱 Mobile Market Dashboard"

app.layout = html.Div([
    html.H1("📱 Mobile Market Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Model:"),
        dcc.Dropdown(
            options=[{'label': model, 'value': model} for model in sorted(df['Model Name'].unique())],
            id='model-dropdown',
            placeholder="Filter by model",
            multi=True
        )
    ], style={'width': '40%', 'margin': 'auto'}),

    dcc.Graph(id='scatter-ram-camera'),
    dcc.Graph(id='scatter-battery-price'),
    dcc.Graph(id='bar-top-models'),
    dcc.Graph(id='line-ram-price'),
    dcc.Graph(id='line-year-camera'),
    dcc.Graph(id='line-battery-price'),

    html.H3("📋 Full Dataset Overview"),
    dcc.Graph(
        id='data-table',
        figure=px.scatter(df, x='RAM', y='nched Price (USA)', hover_name='Model Name')
    )
])

@app.callback(
    [Output('scatter-ram-camera', 'figure'),
     Output('scatter-battery-price', 'figure'),
     Output('bar-top-models', 'figure'),
     Output('line-ram-price', 'figure'),
     Output('line-year-camera', 'figure'),
     Output('line-battery-price', 'figure')],
    [Input('model-dropdown', 'value')]
)
def update_graphs(selected_models):
    filtered_df = df[df['Model Name'].isin(selected_models)] if selected_models else df

    fig1 = px.scatter(
        filtered_df, x='RAM', y='Main Back Camera',
        color='Customer Segment', hover_name='Model Name',
        title='Customer Segments: RAM vs Main Camera',
        labels={'RAM': 'RAM (GB)', 'Main Back Camera': 'Back Cam (MP)'}
    )

    fig2 = px.scatter(
        filtered_df, x='Battery Capacity', y='nched Price (USA)',
        color='Customer Segment', hover_name='Model Name',
        title='Customer Segments: Battery vs Price',
        labels={'Battery Capacity': 'Battery (mAh)', 'nched Price (USA)': 'Price (USD)'}
    )

    top_models = (filtered_df.groupby('Model Name')['nched Price (USA)']
                  .mean().sort_values(ascending=False).head(10).reset_index())

    fig3 = px.bar(
        top_models, x='Model Name', y='nched Price (USA)',
        title='Top 10 Most Expensive Models (USD)',
        labels={'nched Price (USA)': 'Avg Price (USD)'},
        color='nched Price (USA)', color_continuous_scale='Reds'
    )

    fig4 = px.line(
        filtered_df.sort_values('RAM'), x='RAM', y='nched Price (USA)',
        title='Price Distribution by RAM',
        labels={'RAM': 'RAM (GB)', 'nched Price (USA)': 'Price (USD)'}
    )

    fig5 = px.line(
        filtered_df.groupby('Launched Year')['Main Back Camera'].max().reset_index(),
        x='Launched Year', y='Main Back Camera',
        title='Max Camera Resolution Over Time',
        markers=True
    )

    fig6 = px.line(
        filtered_df.sort_values('Battery Capacity'), x='Battery Capacity', y='nched Price (USA)',
        title='Battery Capacity vs. Price',
        labels={'Battery Capacity': 'Battery (mAh)', 'nched Price (USA)': 'Price (USD)'}
    )

    return fig1, fig2, fig3, fig4, fig5, fig6

if __name__ == '__main__':
    app.run(debug=True)
