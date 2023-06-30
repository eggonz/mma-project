import base64
import io
import os
import re
from typing import Any

import dash.exceptions
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from dash import Dash, Input, Output, State, callback, ctx, dash_table, dcc, html, no_update
from dash_extensions.javascript import Namespace, arrow_function

import gpt
from clip import load_clip_model, compute_emb_distances
from dataset import FundaPrecomputedEmbeddings

# ASSETS_PATH = os.getenv("ASSETS_PATH")
# ADS_PATH = os.getenv("ADS_PATH")
# EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH")
# IMAGES_PATH = os.getenv("IMAGES_PATH")


ASSETS_PATH = "/home/miranda/Documents/school/UvA/6_MMA/mma-project/assets/"
ADS_PATH = "./data/final.pkl"
EMBEDDINGS_PATH = "./data/clip_embeddings/funda_images_tiny.pkl"
IMAGES_PATH = "../Datasets/Funda/images"
# Comment this line if you want to use true GPT (need API key)
# gpt.get_pandas_query = gpt.get_pretty_prompt = lambda x: x

# #
# --------------- Reading in some initial data ----------------
# #

df = pd.read_pickle(ADS_PATH)
df["city"] = df["city"].str.lower()

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
colorway=['#001219','#005F73','#0A9396','#94D2BD','#E9D8A6','#EE9B00','#CA6702','#BB3E03','#AE2012', '#9B2226']

def create_table_df(houses):
    houses_ids = houses["funda"].unique()

    filtered_embeddings = ClipStuff.dataset_clip_embeddings.filter_ids(houses_ids)

    if ClipStuff.query_clip_embedding is None:
        ClipStuff.ranking = np.zeros(len(filtered_embeddings))

    ranked_houses = filtered_embeddings.rank_houses(ClipStuff.ranking)

    return (
        houses.loc[
            ranked_houses,
            [
                "city",
                "price",
                "living_area_size",
                "nr_bedrooms",
                "energy_label"
            ],
        ]
        .reset_index(drop=True)
        .to_dict("records")
    )


def create_umap(houses):
    houses_ids = houses["funda"].unique()

    filtered_embeddings = ClipStuff.dataset_clip_embeddings.filter_ids(houses_ids)
    umap_coords = filtered_embeddings.get_all_umaps()

    if ClipStuff.query_clip_embedding is None:
        ClipStuff.ranking = np.zeros(len(filtered_embeddings))
    else:
        ClipStuff.ranking = compute_emb_distances(
            ClipStuff.query_clip_embedding, filtered_embeddings.get_all_embeddings()
        )
    funda_ids = filtered_embeddings.get_all_ids()
    img_paths = filtered_embeddings.df.index.values

    color = np.tanh(ClipStuff.ranking / 0.1)
    data = pd.DataFrame(
        {
            "umapx": umap_coords[:, 0],
            "umapy": umap_coords[:, 1],
            "rank": color,
            "funda_id": funda_ids,
            "image_path": img_paths,
        }
    )
    size = np.clip(color * 0.25 + 0.25, 0, 0.5)

    fig = px.scatter(
        data,
        x="umapx",
        y="umapy",
        color="rank",
        size=size,
        custom_data=["funda_id", "image_path"],
        opacity=0.5,
    )

    fig.update_layout(
        plot_bgcolor = '#f0f1ee',
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis={"visible": False, "showticklabels": False},
        yaxis={"visible": False, "showticklabels": False},
    )
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    return fig


def create_histo(data):
    fig = px.histogram(data, x="price", nbins=20, labels={'price': 'price (€)'}, color_discrete_sequence=['#005F73'])
    fig.update_layout(
        plot_bgcolor = '#f0f1ee',
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    return fig


labels = dict(
    zip(
        range(1, 14),
        [
            "A+++++",
            "A++++",
            "A+++",
            "A++",
            "A+",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "N/A",
        ],
    )
)


def create_pie(data):
    pie_df = data["energy_label"].value_counts().reset_index()
    pie_df["label"] = pie_df["energy_label"].astype(int).map(labels)
    fig = px.pie(
        pie_df,
        names="label",
        values="count",
        color_discrete_sequence=colorway
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
        labels={'price': 'price (€)', 'living_area_size': 'area (m^2)'},
        custom_data=["funda"],
        color_discrete_sequence=['#005F73']
    )
    fig.update_layout(
        plot_bgcolor = '#f0f1ee',
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
    "stack": [{"df": df, "prompt": "", "subset": None}],
    "active_id": None,
    "children": None,
}


def current() -> pd.DataFrame:
    curr = state["stack"][-1]
    return curr["df"] if curr["subset"] is None else curr["subset"]


figures: dict[str, Any] = {
    "geomap": None,
    "umap": None,
    "histo": None,
    "pie": None,
    "scatter": None,
    "table": None,
}


class ClipStuff:
    model = load_clip_model()
    query_clip_embedding = None
    dataset_clip_embeddings = FundaPrecomputedEmbeddings.from_pickle(EMBEDDINGS_PATH)
    ranking = None

    @staticmethod
    def reset():
        ClipStuff.query_clip_embedding = None
        ClipStuff.ranking = None


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
    figures["table"] = create_table_df(houses)

    return (
        figures["umap"],
        figures["histo"],
        figures["pie"],
        figures["scatter"],
        figures["table"],
    )


def reduce_houses(text_prompt):
    # ask gpt
    filter_prompt = gpt.get_pandas_query(text_prompt)
    pretty_prompt = gpt.get_pretty_prompt(filter_prompt)

    print(filter_prompt)
    print(pretty_prompt)

    # update state
    if pretty_prompt in {p["prompt"] for p in state["stack"]}:
        return

    houses = state["stack"][-1]["df"].query(filter_prompt)
    state["stack"].append({"df": houses, "prompt": pretty_prompt, "subset": None})

    update_plots(houses)
    return houses


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
                style={"width": "100%", "height": "100%", "marginTop": "10px"},
            ),
            html.Hr(),
        ]
    )


def get_info(feature=None):
    header = [html.H4("House Attributes")]
    if not feature or ("properties" not in feature) or feature["properties"]["cluster"]:
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


header = html.Div(
    html.H1("FUNDA EXPLORER"),
    className='header'
)

info = html.Div(
    children=get_info(),
    id="info",
    className="info",
    style={
        "position": "absolute",
        "top": "10px",
        "right": "10px",
        "zIndex": "1000",
        "color": "black",
        "pointerEvents": "none",
    },
)

ns = Namespace("myNamespace", "mySubNamespace")
map = html.Div(
    [
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("Map of Houses", className="card-title"),
                        html.P(
                            "Location of the houses in the geographic map.",
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
                                    zoomToBounds=False,
                                    superClusterOptions={"radius": 100, "minPoints": 5},
                                    options=dict(pointToLayer=ns("pointToLayer")),
                                    hoverStyle=arrow_function(
                                        dict(weight=5, color="red", dashArray="")
                                    ),
                                ),
                                info,
                            ],
                            center=(52.1326, 5.2913),
                            zoom=7,
                            style={
                                "width": "100%",
                                "height": "50vh",
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
                        html.H4("Similarity Exploration", className="card-title"),
                        html.P(
                            "Space to explore houses with similar styles.",
                            className="card-text",
                        ),
                        dcc.Graph(
                            id="umap",
                            figure=figures["umap"],
                            style={"width": "100%", "height": "60%"},
                            clear_on_unhover=True
                        ),
                        dcc.Tooltip(id="umap-tooltip"),
                    ]
                ),
            ]
        )
    ]
)

table = html.Div(
    [
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H4("Ranked houses", className="card-title"),
                        dash_table.DataTable(
                            data=figures["table"],
                            id="housetable",
                            page_current=0,
                            page_size=6,
                            style_as_list_view=True,
                            style_cell={"font-family": "Manrope", "textAlign": "left"},
                            style_header={"backgroundColor": "#A49B98", "fontWeight": "700", "color": "white"}
                        ),
                    ]
                )
            ]
        )
    ]
)

prompt_holders = html.Div(
    [
        dbc.Card(
            [
                dbc.CardHeader("User Input"),
                dbc.CardBody([
                        html.Div(
                            html.Div(
                                [
                                    html.H4("Description prompt:", className="card-title"),
                                    html.P(
                                        "Natural language description of the house requirements that will filter the displayed data.",
                                        className="card-text",
                                    ),
                                    dcc.Textarea(
                                        id="text_prompt",
                                        # type="text",
                                        # debounce=True,
                                        placeholder="I want a house in Amsterdam or in Rotterdam",
                                        style={"width": "100%"},
                                    ),
                                    html.Div(id="previous_prompts"),
                                ]
                            ),
                            style={"width": "100%", "marginBottom": "20px"},
                        ),

                        html.Div(
                            html.Div(
                                [
                                    dbc.Button(
                                        "Reset",
                                        id="reset_button",
                                        n_clicks=0,
                                        style={"marginRight": "5px"},
                                    ),
                                    dbc.Button("Undo", id="undo_button", n_clicks=0),
                                ]
                            ),
                            style={"marginTop": "10px"},
                        ),
                        html.Hr(),
                        html.Div(
                            [
                                html.H4("Image for similarity:", className="card-title"),
                                html.P(
                                    "You can provide an image of an example house that is similar to the idea you have in mind. "
                                    "This image will be used to rank the results based on similarity.",
                                    className="card-text",
                                ),
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
                                        "border-color": "#A49B98",
                                        "textAlign": "center",
                                        "margin": "0px",
                                        "background": "#f0f1ee"
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="output-image-upload"),
                            ]
                        ),
                    ]
                )
            ]
        )
    ],
    style={"marginLeft": "10px"},
)

map_meta_data = dbc.Stack([map, table])

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

items = ["data:image/png;base64,{}".format(img.decode()) for key,img in enumerate(encoded_image)]
image_table_holders = html.Div([
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Images of the selected house",
                                style={"marginLeft": "10px", "marginBottom": "20px"}),
                    dbc.Carousel(
                    items = [{"key": key, "src": item} for key,item in enumerate(items)],
                    controls = True,
                    indicators=True,
                    id = 'image_container'
                    )
                ]
            )
        )
    ], style = {"marginBottom": "20px"}
)

image_umap = html.Div([image_table_holders, umap])

mini_plot_holder = dbc.Stack(
    [
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Prices", className="card-title"),
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
                    html.H6("Energy Labels", className="card-title"),
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
                        "Living space vs. Price", className="card-title"
                    ),
                    dcc.Graph(
                        id="scatter",
                        figure=figures["scatter"],
                        style={"width": "100%", "height": "24vh"},
                    ),
                ]
            )
        ),
    ], style={"margin-right":"1rem"}
)

app_main = html.Div(
    [
        header,
        html.Div([
            dbc.Row(
                    [
                        dbc.Col(prompt_holders, align="start", width=2),
                        dbc.Col(map_meta_data, align="start", width=4),
                        dbc.Col(image_umap, align="start", width=4),
                        dbc.Col(mini_plot_holder, align="start", width=2),
                    ],
                    align="center")
        ])
    ]
)


app.layout = app_main

# #
# --------------- Callbacks for interactions ----------------
# #

@callback(
    [
        Output("mapstate", "center", allow_duplicate=True),
        Output("mapstate", "zoom", allow_duplicate=True),
    ],
    Input("scatter", "clickData"),
    prevent_initial_call=True
)
def on_click_scatter(scatter):
    if scatter is not None and ctx.triggered_id == "scatter":
        funda = scatter["points"][0]["customdata"][0]
    lat,lon = df.loc[df["funda"] == funda, "lat"].iloc[0], df.loc[df["funda"] == funda, "lon"].iloc[0]
    coordinates = (lat,lon)
    return [coordinates, 17]

@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("histo", "figure", allow_duplicate=True),
        Output("pie", "figure", allow_duplicate=True),
        Output("scatter", "figure", allow_duplicate=True),
        Output("housetable", "data", allow_duplicate=True),
    ],
    Input("mapstate", "bounds"),
    prevent_initial_call=True,
)
def on_pan_geomap(bounds):
    ((lat_min, lon_min), (lat_max, lon_max)) = bounds

    df = state["stack"][-1]["df"]
    subset = df.loc[
        df["lat"].between(lat_min, lat_max) & df["lon"].between(lon_min, lon_max)
    ]
    state["stack"][-1]["subset"] = subset
    return update_plots(subset)


@callback(
    [
        Output("mapstate", "center", allow_duplicate=True),
        Output("mapstate", "zoom", allow_duplicate=True),
    ],
    Input("umap", "clickData"),
    prevent_initial_call=True,
)
def on_click_umap(umap):
    if umap is not None and ctx.triggered_id == "umap":
        funda = umap["points"][0]["customdata"][0]
    lat, lon = (
        df.loc[df["funda"] == funda, "lat"].iloc[0],
        df.loc[df["funda"] == funda, "lon"].iloc[0],
    )
    coordinates = (lat, lon)
    return [coordinates, 17]


@callback(
    [
        Output("output-image-upload", "children", allow_duplicate=True),
        Output("umap", "figure", allow_duplicate=True),
        Output("housetable", "data", allow_duplicate=True),
    ],
    Input("upload-image", "contents"),
    State("upload-image", "filename"),
    prevent_initial_call=True,
)
def update_uploaded_image(content, name):
    """
    it replaces the content of output-image-upload with the uploaded image and its name

    content: image encoded in str, base64 encoding of the image
    name: name of the uploaded file
    """
    if content is not None:
        clean_content = re.sub("^data:image/.+;base64,", "", content)
        decoded_bytes = base64.b64decode(clean_content)
        image_bytes = io.BytesIO(decoded_bytes)
        img = Image.open(image_bytes)

        ClipStuff.query_clip_embedding = ClipStuff.model.get_visual_emb_for_img(img)

        children = [parse_contents(c, n) for c, n in zip([content], [name])]

        curr = current()
        figures["umap"] = create_umap(curr)
        figures["table"] = create_table_df(curr)
        return children, figures["umap"], figures["table"]

    ClipStuff.reset()


@callback(Output("info", "children"), [Input("geomap", "hover_feature")])
def info_hover(feature):
    return get_info(feature)


@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("histo", "figure", allow_duplicate=True),
        Output("pie", "figure", allow_duplicate=True),
        Output("scatter", "figure", allow_duplicate=True),
        Output("housetable", "data", allow_duplicate=True),
        Output("geomap", "data", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True),
    ],
    Input("text_prompt", "value"),
    prevent_initial_call=True,
)
def on_input_prompt(text_prompt):
    if not isinstance(text_prompt, str) or len(text_prompt) < 15:
        raise dash.exceptions.PreventUpdate

    reduced_points = reduce_houses(text_prompt)
    prompt_list = update_prompt_list()
    geomap = dlx.dicts_to_geojson(reduced_points.to_dict("records"))

    umap, histo, pie, scatter, housetable = update_plots(reduced_points)

    return (umap, histo, pie, scatter, housetable, geomap, prompt_list)


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
        raise dash.exceptions.PreventUpdate

    undo()
    prompt_list = update_prompt_list()
    points = dlx.dicts_to_geojson(state["stack"][-1]["df"].to_dict("records"))

    return figures["umap"], points, prompt_list


@callback(
    [
        Output("umap", "figure", allow_duplicate=True),
        Output("geomap", "data", allow_duplicate=True),
        Output("previous_prompts", "children", allow_duplicate=True),
        Output("output-image-upload", "children", allow_duplicate=True),
    ],
    Input("reset_button", "n_clicks"),
    prevent_initial_call=True,
)
def on_button_reset(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    reset()
    prompt_list = update_prompt_list()
    points = dlx.dicts_to_geojson(df.to_dict("records"))

    ClipStuff.reset()
    figures["umap"] = create_umap(state["stack"][-1]["df"])

    return figures["umap"], points, prompt_list, []


@callback(
    [
        Output("umap-tooltip", "show"),
        Output("umap-tooltip", "bbox"),
        Output("umap-tooltip", "children"),
    ],
    Input("umap", "hoverData"),
    prevent_initial_call=True,
)
def on_umap_hover(umap):
    if umap is None:
        return False, no_update, no_update
    if umap is not None and ctx.triggered_id == "umap":
        bbox, img = umap["points"][0]["bbox"], umap["points"][0]["customdata"][1]

    encoded_image = base64.b64encode(open(f"{IMAGES_PATH}/{img}", "rb").read())
    children = [
        html.Div(
            [
                html.Img(
                    src="data:image/png;base64,{}".format(encoded_image.decode()),
                    style={"height": "100%", "width": "100%"},
                )
            ],
            style={"width": "100px", "white-space": "normal"},
        )
    ]
    return [True, bbox, children]


@callback(
    [Output("image_container", "items")],
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
        raise dash.exceptions.PreventUpdate

    # Use the latest dataframe to get the on hover information.
    df = current()
    imgs = df.loc[df["funda"] == funda, "images_paths"]
    encoded_image = [
        base64.b64encode(open(f"{IMAGES_PATH}/{img}", "rb").read())
        for img in imgs.iloc[0]
    ]
    items = [{"key": key, "src": "data:image/png;base64,{}".format(img.decode())} for key,img in enumerate(encoded_image)]

    return [items]


# @app.callback(Output("image-container", "children"), [Input("image-slider", "value")])
# def update_image(value):
#     if value is None:
#         return

#     image_path = state["children"][int(value)]
#     return image_path


# #
# --------------- Main ----------------
# #

if __name__ == "__main__":
    app.run_server(debug=True)
