import base64
import json
import os

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
import pandas as pd
import plotly.express as px
import requests
from dash_extensions.javascript import arrow_function
from PIL import Image
from dash import Dash, Input, State, Output, callback, ctx, dash_table, dcc, html

import gpt

ASSETS_PATH = "../assets"
ADS_PATH = "../data/final.pkl"
EMBEDDINGS_PATH = "../data/embeddings.pkl"
IMAGES_PATH = "../funda"

# #
# --------------- Reading in some initial data ----------------
# #

df = pd.read_pickle(ADS_PATH)

# print(df[["city", "nr_bedrooms", "energy_label"]])

# #
# --------------- Functions for creating the plotly plots ----------------
# #

# def create_geomap(data):
#     fig = px.scatter_mapbox(data, lat="lat", lon="lon",
#                             zoom=7,
#                             custom_data=['funda'])
#     fig.update_layout(mapbox_style="open-street-map", uirevision=True)
#     fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False)

#     return fig

def create_umap(data):
    fig = px.scatter(data, x="energy_label", y="nr_bedrooms", custom_data=['funda'], width=400, height=400)
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis={'visible': False, 'showticklabels': False},
        yaxis={'visible': False, 'showticklabels': False}
    )

    return fig

# #
# --------------- Global variables ----------------
# #

indices = [
    42194016, 42194023, 42194046, 42194072, 42194086,
    42194088, 42194092, 42194096, 42194098
]
df = df[df['funda'].isin(indices)]

state = {
    "stack": [{"df": df, "prompt": ""}],
    "active_id": None,
    "children": None,
}

figures = {
    "geomap": None,
    "umap": None
}

app = Dash(__name__, assets_folder=ASSETS_PATH, external_stylesheets=[dbc.themes.DARKLY])

# #
# --------------- State changing functions ----------------
# #

def update_maps():
    houses = state["stack"][-1]["df"]
    figures["umap"] = create_umap(houses)

def reduce_houses(text_prompt):

    # ask gpt
    filter_prompt = gpt.get_pandas_query(text_prompt)
    pretty_prompt = gpt.get_pretty_prompt(filter_prompt)

    # update state
    if pretty_prompt in {p["prompt"] for p in state["stack"]}:
        return

    houses = state["stack"][-1]["df"].query(filter_prompt)
    state["stack"].append({"df": houses, "prompt": pretty_prompt})

    update_maps()
    return houses

def update_colors(idx):
    color = ["blue"] * len(state["stack"][-1]["df"])
    if idx is not None:
        color[idx] = "red"

    figures["umap"] = figures["umap"].update_traces(marker=dict(color=color))

def update_prompt_list():
    return  html.Ol([
        html.Li(p["prompt"]) for p in state["stack"][1:]
    ])

def reset():
    if len(state["stack"]) <= 1:
        return

    state["stack"] = state["stack"][:1]
    update_maps()

def undo():
    if len(state["stack"]) <= 1:
        return

    state["stack"] = state["stack"][:-1]
    update_maps()


update_maps()

# #
# --------------- HTML Layout ----------------
# #

def parse_contents(contents, filename):
    return html.Div([
        html.H5(filename),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, style = {"width": "100px", "height": "100px"}),
        html.Hr()])

def get_info(feature=None):
    header = [html.H4("House Attributes")]
    if not feature:
        return header + [html.P("Hoover over on a house")]
    elif feature["properties"]["cluster"]:
        return header + [html.P("Hoover over on a house")]
    else:
        return header + ["€ {}".format(feature["properties"]["price"]), html.Br(),
                     "{} ㎡".format(feature["properties"]["living_area_size"]), html.Br(),
                     "{} € per ㎡".format(feature["properties"]["price_per_m2"]), html.Br(),
                     "{} number of rooms".format(feature["properties"]["nr_rooms"]), html.Br(),
                     "{} number of bedrooms".format(feature["properties"]["nr_bedrooms"])]


info = html.Div(children=get_info(), id="info", className="info",
                style={"position": "absolute", "top": "10px", "right": "10px", "z-index": "1000", "color": "black", "pointer-events":"none"})


map = dbc.Card([
    dbc.CardBody([
        html.H4("Map of Houses", className="card-title"),
        html.P(
            "Some quick example text to build on the card title and "
            "make up the bulk of the card's content.",
            className="card-text",
        ),
        dl.Map([
            dl.TileLayer(),
            dl.GeoJSON(
                data=dlx.dicts_to_geojson(df.to_dict('records')),
                cluster=True,
                id="geomap",
                zoomToBoundsOnClick=True,
                zoomToBounds=True,
                superClusterOptions={"radius": 100},
            ),
        info
        ], center=(52.1326, 5.2913), zoom=7, style={
            'width': '100%',
            'height': '50vh',
            'margin': "auto",
            "display": "block"
        }),
    ], className="content"),
])

umap = dbc.Card([
    dbc.CardBody([
        html.H4("Map of Houses", className="card-title"),
        html.P(
            "Some quick example text to build on the card title and "
            "make up the bulk of the card's content.",
            className="card-text",
        ),
        dcc.Graph(
            id="umap",
            figure=figures["umap"]
        )
    ], className="content"),
])


prompt_holders = html.Div([
    html.Div(
        html.Div([
            html.H3("Enter prompt here:"),
            dcc.Input(
                id="text_prompt", type="text", debounce=True,
                placeholder="I want a house in Amsterdam or in Rotterdam"
            ),
            html.Div(id="previous_prompts")
        ]),
    ),
    html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Upload an image ',
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload'),
    ]),
    html.Div(
        html.Div([
            html.Button('Reset', id='reset_button', n_clicks=0),
            html.Button('Undo', id='undo_button', n_clicks=0)
        ]),
        style={"margin-top": "10px"}
    )
])

map_holders = [map, umap]

image_table_holders = html.Div([
    html.H3("Images of the clicked house", style= {"margin-left": "10px"}),
    html.Div(id="image_container", className="content"),
]),

app_main = dbc.Container([
    dbc.Row([
        dbc.Col(prompt_holders, width= 2),
        dbc.Col(map_holders, width = 6, style = {"margin-left": "30px"}),
        dbc.Col(image_table_holders, width= 3),
    ],  style={'height': '100%'}),
], fluid=True)


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "5rem",
    "padding": "2rem 1rem",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "12rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.P(
            "Change from exploration to analysis", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Exploration", href="/", active="exact"),
                dbc.NavLink("Analysis", href="/analysis", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content
])

# #
# --------------- Callbacks for interactions ----------------
# #


@callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip([list_of_contents], [list_of_names])]
        return children
    

@callback(Output("info", "children"), [Input("geomap", "hover_feature")])
def info_hover(feature):
    return get_info(feature)



@callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return app_main
    elif pathname == "/analysis":
        return html.P("This is the content of page 1. Yay!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("geomap", "data", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True)
    ],
    Input("text_prompt", "value"),
    prevent_initial_call=True
)
def on_input_prompt(text_prompt):
    if text_prompt is None:
        return
    reduce_houses(text_prompt)
    prompt_list = update_prompt_list()

    reduced_points = state["stack"][-1]["df"].query(text_prompt)
    reduced_points = dlx.dicts_to_geojson(reduced_points.to_dict('records'))

    return (
        figures["umap"],
        reduced_points,
        prompt_list
    )

@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("geomap", "data", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True)
    ],
    Input("undo_button", "n_clicks"),
    prevent_initial_call=True
)
def on_button_undo(n_clicks):
    if n_clicks is None:
        return

    undo()
    prompt_list = update_prompt_list()
    points = dlx.dicts_to_geojson(state["stack"][-1]["df"].to_dict('records'))

    return figures["umap"], points, prompt_list

@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("geomap", "data", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True)
    ],
    Input("reset_button", "n_clicks"),
    prevent_initial_call=True
)
def on_button_reset(n_clicks):
    if n_clicks is None:
        return

    reset()
    prompt_list = update_prompt_list()
    points = dlx.dicts_to_geojson(df.to_dict('records'))

    return figures["umap"], points, prompt_list

@callback(
    [
        Output("umap", "figure"),
        Output("image_container", "children")
    ],
    Input("geomap", "click_feature"),
    Input("umap", "clickData"),
    prevent_initial_call=True
)
def on_hover(geo, umap):
    idx = None
    if geo is not None and ctx.triggered_id == "geomap":
        isCluster = geo['properties']['cluster']
        if not isCluster:
            funda, idx = geo['properties']['funda'], 0

    if umap is not None and ctx.triggered_id == "umap":
        funda, idx = umap['points'][0]["customdata"][0], umap["points"][0]["pointIndex"]

    if idx is None:
        return None

    update_colors(idx=idx)

    # Use the latest dataframe to get the on hover information.
    df = state["stack"][-1]["df"]
    imgs = df.loc[df["funda"] == funda, 'images_paths']
    encoded_image = [
        base64.b64encode(open(f'{IMAGES_PATH}/{img}', 'rb').read())
        for img in imgs.iloc[0]
    ]
    children = [
        html.Img(
            src='data:image/png;base64,{}'.format(img.decode()),
            style={"height": "200px", "width" : "200px", "margin-left": "20px"}
        )
        for img in encoded_image
    ]
    state["children"] = children

    slider = html.Div([
        dcc.Slider(
            id='image-slider',
            min=0,
            max=len(children) - 1,
            value=0,
            step = 1,
            marks={i: str(i + 1) for i in range(len(children))}
        ),
        html.Div(id='image-container')
    ])

    return [figures['umap'], slider]

@app.callback(Output('image-container', 'children'), [Input('image-slider', 'value')])
def update_image(value):
    if value is None:
        return

    image_path = state["children"][int(value)]
    return image_path

@callback(
    Output("image_prompt_vis", "style"),
    Input("image_prompt", "value")
)
def on_input_image_prompt(url):
    if url is None:
        return

    r = requests.get(url, stream=True)
    img = Image.open(r.raw)

    return {"background-image": f"url({url})"}

# #
# --------------- Main ----------------
# #

if __name__ == '__main__':
    app.run_server(debug=True)
