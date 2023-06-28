import base64
import os

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
import pandas as pd
import plotly.express as px
import requests
from PIL import Image

import gpt
from dash import (Dash, Input, Output, State, callback, ctx, dash_table, dcc,
                  html)

ASSETS_PATH = os.getenv("ASSETS_PATH")
ADS_PATH = os.getenv("ADS_PATH")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH")
IMAGES_PATH = os.getenv("IMAGES_PATH")

# Comment this line if you want to use true GPT (need API key)
gpt.get_pandas_query = gpt.get_pretty_prompt = lambda x: x

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
    fig = px.scatter(
        data,
        x="energy_label",
        y="nr_bedrooms",
        custom_data=["funda"],
        width=400,
        height=400,
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis={"visible": False, "showticklabels": False},
        yaxis={"visible": False, "showticklabels": False},
    )

    return fig


def create_histo(data):
    fig = px.histogram(data, x="price", nbins=20)
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    return fig


def create_pie(data):
    fig = px.pie(
        data["energy_label"].value_counts().reset_index(),
        names="energy_label",
        values="count",
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    return fig


def create_scatter(data):
    fig = px.scatter(
        data,
        x="price",
        y="living_area_size",
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    return fig


# #
# --------------- Global variables ----------------
# #

# indices = [
#     42194016, 42194023, 42194046, 42194072, 42194086,
#     42194088, 42194092, 42194096, 42194098
# ]
# df = df[df['funda'].isin(indices)]

state = {
    "stack": [{"df": df, "prompt": ""}],
    "active_id": None,
    "children": None,
}

figures = {"geomap": None, "umap": None, "histo": None, "pie": None, "scatter": None}

app = Dash(
    __name__, assets_folder=ASSETS_PATH, external_stylesheets=[dbc.themes.LITERA]
)

# #
# --------------- State changing functions ----------------
# #


def update_plots(houses):
    figures["umap"] = create_umap(houses)
    figures["histo"] = create_histo(houses)
    figures["pie"] = create_pie(houses)
    figures["scatter"] = create_scatter(houses)

    return (figures["umap"], figures["histo"], figures["pie"], figures["scatter"])


def reduce_houses(text_prompt):
    # ask gpt
    filter_prompt = gpt.get_pandas_query(text_prompt)
    pretty_prompt = gpt.get_pretty_prompt(filter_prompt)

    # update state
    if pretty_prompt in {p["prompt"] for p in state["stack"]}:
        return

    houses = state["stack"][-1]["df"].query(filter_prompt)
    state["stack"].append({"df": houses, "prompt": pretty_prompt})

    update_plots(houses)
    return houses


def update_colors(idx):
    color = ["blue"] * len(state["stack"][-1]["df"])
    if idx is not None:
        color[idx] = "red"

    figures["umap"] = figures["umap"].update_traces(marker=dict(color=color))


def update_prompt_list():
    return html.Ol([html.Li(p["prompt"]) for p in state["stack"][1:]])


def reset():
    if len(state["stack"]) <= 1:
        return

    state["stack"] = state["stack"][:1]
    houses = state["stack"][-1]["df"]
    update_plots(houses)


def undo():
    if len(state["stack"]) <= 1:
        return

    state["stack"] = state["stack"][:-1]
    houses = state["stack"][-1]["df"]
    update_plots(houses)


update_plots(df)

# #
# --------------- HTML Layout ----------------
# #


def parse_contents(contents, filename):
    return html.Div(
        [
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            html.Img(
                src=contents,
                style={"width": "100%", "height": "100%", "margin-top": "10px"},
            ),
            html.Hr(),
        ]
    )


def get_info(feature=None):
    header = [html.H4("House Attributes")]
    if not feature:
        return header + [html.P("Hover over a house")]
    elif feature["properties"]["cluster"]:
        return header + [html.P("Hover over a house")]
    else:
        return header + [
            "€ {}".format(feature["properties"]["price"]),
            html.Br(),
            "{} ㎡".format(feature["properties"]["living_area_size"]),
            html.Br(),
            "{} € per ㎡".format(feature["properties"]["price_per_m2"]),
            html.Br(),
            "{} rooms".format(feature["properties"]["nr_rooms"]),
            html.Br(),
            "{} bedrooms".format(feature["properties"]["nr_bedrooms"]),
        ]


info = html.Div(
    children=get_info(),
    id="info",
    className="info",
    style={
        "position": "absolute",
        "top": "10px",
        "right": "10px",
        "z-index": "1000",
        "color": "black",
        "pointer-events": "none",
    },
)


map = html.Div(
    [
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("Map of Houses", className="card-title"),
                        html.P(
                            "Some quick example text to build on the card title and "
                            "make up the bulk of the card's content.",
                            className="card-text",
                        ),
                        dl.Map(
                            [
                                dl.TileLayer(),
                                dl.GeoJSON(
                                    data=dlx.dicts_to_geojson(df.to_dict("records")),
                                    cluster=True,
                                    id="geomap",
                                    zoomToBoundsOnClick=True,
                                    zoomToBounds=True,
                                    superClusterOptions={"radius": 100},
                                ),
                                info,
                            ],
                            center=(52.1326, 5.2913),
                            zoom=7,
                            style={
                                "width": "100%",
                                "height": "70vh",
                                "margin": "auto",
                                "display": "block",
                            },
                            id="mapstate",
                        ),
                    ]
                ),
            ]
        )
    ]
)

umap = html.Div(
    [
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("UMAP of Image embeddings", className="card-title"),
                        dcc.Graph(
                            id="umap",
                            figure=figures["umap"],
                            style={"width": "100%", "height": "60%"},
                        ),
                    ]
                ),
            ]
        )
    ]
)


prompt_holders = html.Div(
    [
        html.Div(
            html.Div(
                [
                    html.H4("GPT Prompt:"),
                    dcc.Input(
                        id="text_prompt",
                        type="text",
                        debounce=True,
                        placeholder="I want a house in Amsterdam or in Rotterdam",
                        style={"width": "100%"},
                    ),
                    html.Div(id="previous_prompts"),
                ]
            ),
            style={"width": "100%", "margin-bottom": "20px"},
        ),
        html.Div(
            [
                dcc.Upload(
                    id="upload-image",
                    children=html.Div(
                        [
                            "Upload an image ",
                        ]
                    ),
                    style={
                        "width": "100%",
                        "height": "70px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "0px",
                    },
                    multiple=False,
                ),
                html.Div(id="output-image-upload"),
            ]
        ),
        html.Div(
            html.Div(
                [
                    dbc.Button(
                        "Reset",
                        id="reset_button",
                        n_clicks=0,
                        style={"margin-right": "5px"},
                    ),
                    dbc.Button("Undo", id="undo_button", n_clicks=0),
                ]
            ),
            style={"margin-top": "10px"},
        ),
    ],
    style={"margin-left": "10px"},
)

map_meta_data = dbc.Stack([map, html.Div("Meta Data Comes Here")])

imgs_placeholder = [
    "42194016/image1.jpeg",
    "42194016/image2.jpeg",
    "42194016/image3.jpeg",
    "42194016/image4.jpeg",
    "42194016/image5.jpeg",
]
encoded_image = [
    base64.b64encode(open(f"{IMAGES_PATH}/{img}", "rb").read())
    for img in imgs_placeholder
]
children = [
    html.Img(
        src="data:image/png;base64,{}".format(img.decode()),
        style={"height": "100%", "width": "100%"},
    )
    for img in encoded_image
]
state["children"] = children
slider = html.Div(
    [
        dcc.Slider(
            id="image-slider",
            min=0,
            max=len(children) - 1,
            value=0,
            step=1,
            marks={i: str(i + 1) for i in range(len(children))},
        ),
        html.Div(id="image-container"),
    ],
    style={"width": "100%", "height": "100%"},
)

image_table_holders = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Images of the clicked house", style={"margin-left": "10px"}),
            html.Div(id="image_container", children=slider),
        ]
    )
)

image_umap = html.Div([image_table_holders, umap])

mini_plot_holder = dbc.Stack(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Histogram of Prices", className="card-title"),
                    dcc.Graph(
                        id="histo",
                        figure=figures["histo"],
                        style={"width": "100%", "height": "24vh"},
                    ),
                ]
            )
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Pie Chart of Energy Labels", className="card-title"),
                    dcc.Graph(
                        id="pie",
                        figure=figures["pie"],
                        style={"width": "100%", "height": "24vh"},
                    ),
                ]
            )
        ),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6(
                        "Scatterplot of Prices and Living Space", className="card-title"
                    ),
                    dcc.Graph(
                        id="scatter",
                        figure=figures["scatter"],
                        style={"width": "100%", "height": "24vh"},
                    ),
                ]
            )
        ),
    ]
)

app_main = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(prompt_holders, align="start", width=2),
                dbc.Col(map_meta_data, align="start", width=4),
                dbc.Col(image_umap, align="start", width=4),
                dbc.Col(mini_plot_holder, align="start", width=2),
            ],
            align="center",
        )
    ]
)


app.layout = app_main

# #
# --------------- Callbacks for interactions ----------------
# #


@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("histo", "figure", allow_duplicate=True),
        Output("pie", "figure", allow_duplicate=True),
        Output("scatter", "figure", allow_duplicate=True),
    ],
    Input("mapstate", "bounds"),
    prevent_initial_call=True,
)
def on_pan_geomap(bounds):
    ((lat_min, lon_min), (lat_max, lon_max)) = bounds

    print(bounds)

    df = state["stack"][-1]["df"]
    subset = df.loc[
        df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)
    ]
    return update_plots(subset)


@callback(
    Output("output-image-upload", "children"),
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is None:
        return

    children = [
        parse_contents(c, n) for c, n in zip([list_of_contents], [list_of_names])
    ]
    return children


@callback(Output("info", "children"), [Input("geomap", "hover_feature")])
def info_hover(feature):
    return get_info(feature)


@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("histo", "figure", allow_duplicate=True),
        Output("pie", "figure", allow_duplicate=True),
        Output("scatter", "figure", allow_duplicate=True),
        Output("geomap", "data", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True),
    ],
    Input("text_prompt", "value"),
    prevent_initial_call=True,
)
def on_input_prompt(text_prompt):
    if text_prompt is None:
        return
    reduce_houses(text_prompt)
    prompt_list = update_prompt_list()

    reduced_points = state["stack"][-1]["df"].query(text_prompt)
    geomap = dlx.dicts_to_geojson(reduced_points.to_dict("records"))

    umap, histo, pie, scatter = update_plots(reduced_points)

    return (umap, histo, pie, scatter, geomap, prompt_list)


@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("geomap", "data", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True),
    ],
    Input("undo_button", "n_clicks"),
    prevent_initial_call=True,
)
def on_button_undo(n_clicks):
    if n_clicks is None:
        return

    undo()
    prompt_list = update_prompt_list()
    points = dlx.dicts_to_geojson(state["stack"][-1]["df"].to_dict("records"))

    return figures["umap"], points, prompt_list


@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("geomap", "data", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True),
    ],
    Input("reset_button", "n_clicks"),
    prevent_initial_call=True,
)
def on_button_reset(n_clicks):
    if n_clicks is None:
        return

    reset()
    prompt_list = update_prompt_list()
    points = dlx.dicts_to_geojson(df.to_dict("records"))

    return figures["umap"], points, prompt_list


@callback(
    [Output("umap", "figure"), Output("image_container", "children")],
    Input("geomap", "click_feature"),
    Input("umap", "clickData"),
    prevent_initial_call=True,
)
def on_hover(geo, umap):
    idx = None
    if geo is not None and ctx.triggered_id == "geomap":
        isCluster = geo["properties"]["cluster"]
        if not isCluster:
            funda, idx = geo["properties"]["funda"], 0

    if umap is not None and ctx.triggered_id == "umap":
        funda, idx = umap["points"][0]["customdata"][0], umap["points"][0]["pointIndex"]

    if idx is None:
        return None

    update_colors(idx=idx)

    # Use the latest dataframe to get the on hover information.
    df = state["stack"][-1]["df"]
    imgs = df.loc[df["funda"] == funda, "images_paths"]
    encoded_image = [
        base64.b64encode(open(f"{IMAGES_PATH}/{img}", "rb").read())
        for img in imgs.iloc[0]
    ]
    children = [
        html.Img(
            src="data:image/png;base64,{}".format(img.decode()),
            style={"height": "100%", "width": "100%"},
        )
        for img in encoded_image
    ]
    state["children"] = children

    slider = html.Div(
        [
            dcc.Slider(
                id="image-slider",
                min=0,
                max=len(children) - 1,
                value=0,
                step=1,
                marks={i: str(i + 1) for i in range(len(children))},
            ),
            html.Div(id="image-container"),
        ],
        style={"width": "100%", "height": "100%"},
    )

    return [figures["umap"], slider]


@app.callback(Output("image-container", "children"), [Input("image-slider", "value")])
def update_image(value):
    if value is None:
        return

    image_path = state["children"][int(value)]
    return image_path


@callback(Output("image_prompt_vis", "style"), Input("image_prompt", "value"))
def on_input_image_prompt(url):
    if url is None:
        return

    r = requests.get(url, stream=True)
    img = Image.open(r.raw)

    return {"background-image": f"url({url})"}


# #
# --------------- Main ----------------
# #

if __name__ == "__main__":
    app.run_server(debug=True)
