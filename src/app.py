import json

import pandas as pd
import plotly.express as px
import requests
from PIL import Image
import base64
import os
from dash import Dash, Input, Output, callback, ctx, dash_table, dcc, html

ASSETS_PATH = "../assets"
ADS_PATH = "../data/ads.jsonlines"
EMBEDDINGS_PATH = "../data/embeddings.pkl"
IMAGES_PATH = "../funda"

# #
# --------------- Reading in some initial data ----------------
# #

with open(ADS_PATH, encoding="utf8") as fp:
    data = fp.readlines()

parsed = [json.loads(d) for d in data]

df = pd.json_normalize(parsed)
labels = dict(zip(['A+++++', 'A++++', 'A+++', 'A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'Not'], range(1, 14)))

df["images"] = df["images_paths"]
df["lat"] = pd.to_numeric(df["geolocation.lat"])
df["lon"] = pd.to_numeric(df["geolocation.lon"])

df["funda"] = pd.to_numeric(df["funda_identifier"])
df["city"] = df["location_part2"].str.split().str[2]
df["price"] = pd.to_numeric(df["price"].str.split().str[1].str.replace(",", ""), errors="coerce")
df["living_area_size"] = df["highlights.living area"].str.split().str[0].str.replace(",", "").astype(float)
df["nr_bedrooms"] = df["highlights.bedrooms"].astype(float)
df["energy_label"] = df["features.energy.energy label"].str.split().str[0].map(labels)


# print(df[["city", "nr_bedrooms", "energy_label"]])

# #
# --------------- Functions for creating the plotly plots ----------------
# #

def create_geomap(data):
    fig = px.scatter_mapbox(data, lat="lat", lon="lon", 
                            zoom=7, 
                            custom_data=['funda'])
    fig.update_layout(mapbox_style="open-street-map", uirevision=True)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False)
    return fig


def create_umap(data):
    fig = px.scatter(data, x="energy_label", y="nr_bedrooms", custom_data = ['funda'])
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis={'visible': False, 'showticklabels': False},
        yaxis={'visible': False, 'showticklabels': False}
    )
    print(data.shape)
    return fig


# #
# --------------- Global variables ----------------
# #

df = df[[
    "funda", "images", "lat", "lon", "city", "price",
    "living_area_size", "nr_bedrooms", "energy_label"
]]

# indices = [42194016, 42194023, 42194046]
# df = df[df['funda'].isin(indices)]

state = {
    "stack": [{"df": df, "prompt": ""}],
    "active_id": None
}

figures = {
    "geomap": None,
    "umap": None
}

app = Dash(__name__, assets_folder=ASSETS_PATH)


# #
# --------------- State changing functions ----------------
# #

def reduce_houses(prompt):
    print("run1")
    if prompt in {p["prompt"] for p in state["stack"]}:
        return
    print("run2")

    houses = state["stack"][-1]["df"].query(prompt)
    state["stack"].append({"df": houses, "prompt": prompt})

    update_maps()


def update_maps():
    houses = state["stack"][-1]["df"]
    figures["geomap"] = create_geomap(houses)
    figures["umap"] = create_umap(houses)


def update_colors(idx):
    color = ["blue"] * len(state["stack"][-1]["df"])
    if idx is not None:
        color[idx] = "red"
    # color = [
    #     ("red" if i == idx else "blue")
    #     for i in range(len(state["stack"][-1]["df"]))
    # ]

    figures["geomap"] = figures["geomap"].update_traces(marker=dict(color=color))
    figures["umap"] = figures["umap"].update_traces(marker=dict(color=color))


def update_prompt_list():
    return html.Ol([
        html.Li(p["prompt"])
        for p in state["stack"][1:]
    ])


def reset():
    if len(state["stack"]) > 1:
        state["stack"] = state["stack"][:1]
        update_maps()


def undo():
    if len(state["stack"]) > 1:
        state["stack"] = state["stack"][:-1]
        update_maps()


update_maps()

# #
# --------------- HTML Layout ----------------
# #

app.layout = html.Div([
    html.Div([
        html.Div(
            html.Div([
                html.H1("Enter prompt here:"),
                dcc.Input(
                    id="prompt", type="text", debounce=True,
                    placeholder="city == 'Amsterdam' or city == 'Rotterdam'"
                ),
                html.Div(id="previous_prompts")
            ],
                className="content"),
            id="top_left", className="panel"
        ),
        html.Div(
            html.Div([
                html.Div([
                    html.H1("Enter image URL here:"),
                    dcc.Input(
                        id="image_prompt", type="url", debounce=True,
                    )
                ], id="image_inputs"),
                html.Div(id="image_prompt_vis")
            ], className="content"),
            id="middle_left", className="panel"
        ),
        html.Div(
            html.Div([
                html.Button('Reset', id='reset_button', n_clicks=0),
                html.Button('Undo', id='undo_button', n_clicks=0)
            ], id="buttons", className="content"),
            id="bottom_left", className="panel"
        )
    ], id="left"),
    html.Div([
        html.Div(
            html.Div(
                dcc.Graph(
                    id="geomap",
                    figure=figures["geomap"]
                ),
                className="content"
            ),
            id="top_middle", className="panel"
        ),
        html.Div(
            html.Div(
                dcc.Graph(
                    id="umap",
                    figure=figures["umap"]
                ),
                className="content"
            ),
            id="bottom_middle", className="panel"
        )
    ], id="middle"),
    html.Div([
        html.Div(
            html.Div(id="image_container", className="content"),
            id="top_right", className="panel"
        ),
        html.Div(
            html.Div(
                dash_table.DataTable(
                    state["stack"][-1]["df"].drop(columns="images").to_dict("records"),
                    page_size=15, id="data_table"
                ),
                className="content"
            ),
            id="bottom_right", className="panel"
        )
    ], id="right")
], id="container")


# #
# --------------- Callbacks for interactions ----------------
# #

@callback(
    [
        Output("image_container", "children")
    ],
    Input("geomap", "hoverData"),
    Input("umap", "hoverData"),
    prevent_initial_call=True
)
def on_hover(geo, umap):
    idx = None
    if geo is not None and ctx.triggered_id == "geomap":
        # This is the funda identifier
        funda, idx = geo['points'][0]["customdata"][0], geo["points"][0]["pointIndex"]
    if umap is not None and ctx.triggered_id == "umap":
        funda, idx = umap['points'][0]["customdata"][0], umap["points"][0]["pointIndex"]
    if idx is None:
        return None
    update_colors(idx=idx)
    # Use the latest dataframe to get the on hover information.
    imgs = state["stack"][-1]["df"].loc[state["stack"][-1]["df"]["funda"] == funda, 'images'] 
    print(imgs)
    encoded_image = [base64.b64encode(open(f'{IMAGES_PATH}/{img}', 'rb').read()) for img in imgs.iloc[0]]
    children = [html.Img(src='data:image/png;base64,{}'.format(img.decode()),  style={'height':'10%', 'width':'10%'}) for img in encoded_image]   
    return [children]


@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("geomap", "figure", allow_duplicate=True),
        Output("data_table", "data"),
        Output("previous_prompts", "children", allow_duplicate=True)
    ],
    Input("prompt", "value"),
    prevent_initial_call=True
)
def on_input_prompt(prompt):
    if prompt is None:
        return

    reduce_houses(prompt)
    prompt_list = update_prompt_list()
    return (
        figures["umap"],
        figures["geomap"],
        state["stack"][-1]["df"].drop(columns="images").to_dict("records"),
        prompt_list
    )


@callback(
    Output("image_prompt_vis", "style"),
    Input("image_prompt", "value")
)
def on_input_image_prompt(url):
    if url is None:
        return

    print("New image prompt!")

    r = requests.get(url, stream=True)
    img = Image.open(r.raw)

    return {
        "background-image": f"url({url})"
    }


@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("geomap", "figure", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True)
    ],
    Input("reset_button", "n_clicks"),
    prevent_initial_call=True
)
def on_button_reset(n_clicks):
    reset()
    prompt_list = update_prompt_list()
    return figures["umap"], figures["geomap"], prompt_list


@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("geomap", "figure", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True)
    ],
    Input("undo_button", "n_clicks"),
    prevent_initial_call=True
)
def on_button_undo(n_clicks):
    undo()
    prompt_list = update_prompt_list()
    return figures["umap"], figures["geomap"], prompt_list


# #
# --------------- Main ----------------
# #

if __name__ == '__main__':
    app.run_server(debug=True)
