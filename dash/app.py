import json

import pandas as pd
import plotly.express as px

from dash import Dash, Input, Output, callback, ctx, dash_table, dcc, html

# #
# --------------- Reading in some initial data ----------------
# #

with open("./data/ads.jsonlines", encoding="utf8") as fp:
    data = fp.readlines()

parsed = [json.loads(d) for d in data]

df = pd.json_normalize(parsed)
labels = dict(zip(['A+++++', 'A++++', 'A+++', 'A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'Not'], range(1, 14)))

df["images"] = df["images_paths"]
df["lat"] = pd.to_numeric(df["geolocation.lat"])
df["lon"] = pd.to_numeric(df["geolocation.lon"])

df["funda"] = df["funda_identifier"]
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
    fig = px.scatter_mapbox(data, lat="lat", lon="lon", zoom=7)
    fig.update_layout(mapbox_style="open-street-map", uirevision=True)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    return fig

def create_umap(data):
    fig = px.scatter(data, x="energy_label", y="nr_bedrooms")
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        xaxis={'visible': False, 'showticklabels': False},
        yaxis={'visible': False, 'showticklabels': False}
    )
    print(data.shape)
    return fig

# #
# --------------- Global variables ----------------
# #

dfs = {
    "full": df[["funda", "images", "lat", "lon", "city", "price", "living_area_size", "nr_bedrooms", "energy_label"]],
    "small": None
}

def reset():
    dfs["small"] = dfs["full"].copy()

reset()

figures = {
    "geomap": create_geomap(dfs["full"]),
    "umap": create_umap(dfs["full"])
}

app = Dash(__name__)

# #
# --------------- Draw colours on the map ----------------
# #

def update_geo(idx=None):
    color = [
        ("red" if i == idx else "blue")
        for i in range(len(dfs["small"]))
    ]

    print(color)

    return figures["geomap"].update_traces(marker=dict(color=color))

def update_umap(idx=None):
    color = [
        ("red" if i == idx else "blue")
        for i in range(len(dfs["small"]))
    ]

    print(color)

    return figures["umap"].update_traces(marker=dict(color=color))

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
                )
            ],
            className="content"),
            id="top_left", className="panel"
        ),
        html.Div(
            html.Div(className="content"),
            id="middle_left", className="panel"
        ),
        html.Div(
            html.Div(className="content"),
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
                    dfs["small"].drop(columns="images").to_dict("records"),
                    page_size=19
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
    Output("image_container", "style"),
    Input("geomap", "hoverData"),
    Input("umap", "hoverData")
)
def on_hover_geomap_show_image(geo, umap):
    idx = None
    if geo is not None and ctx.triggered_id == "geomap":
        idx = geo["points"][0]["pointIndex"]
    if umap is not None and ctx.triggered_id == "umap":
        idx = umap["points"][0]["pointIndex"]
    if idx is None:
        return None

    imgs = df.loc[idx, "images"]

    return {
        "background-image": f"url(assets/images/{imgs[0]})"
    }

@callback(
    Output("umap", "figure"),
    Input("geomap", "hoverData"),
    Input("umap", "hoverData"),
    Input("prompt", "value")
)
def on_hover_update_umap(geo, umap, prompt):
    if prompt is not None and ctx.triggered_id == "prompt":
        dfs["small"] = dfs["small"].query(prompt)
        figures["umap"] = create_umap(dfs["small"])
        print("RUNNNN")
        return figures["umap"]

    if len(dfs["small"]) > 50:
        return umap

    idx = None
    if geo is not None and ctx.triggered_id == "geomap":
        idx = geo["points"][0]["pointIndex"]
    if umap is not None and ctx.triggered_id == "umap":
        idx = umap["points"][0]["pointIndex"]
    print(idx)
    print(json.dumps(geo, indent=4))
    return update_umap(idx=idx)

@callback(
    Output("geomap", "figure"),
    Input("geomap", "hoverData"),
    Input("umap", "hoverData"),
    Input("prompt", "value")
)
def on_hover_update_geomap(geo, umap, prompt):
    if prompt is not None and ctx.triggered_id == "prompt":

        dfs["small"] = dfs["small"].query(prompt)
        figures["geomap"] = create_geomap(dfs["small"])
        return figures["geomap"]

    if len(dfs["small"]) > 50:
        return figures["geomap"]

    idx = None
    if geo is not None and ctx.triggered_id == "geomap":
        idx = geo["points"][0]["pointIndex"]
    if umap is not None and ctx.triggered_id == "umap":
        idx = umap["points"][0]["pointIndex"]
    print(idx)
    return update_geo(idx=idx)


# #
# --------------- Main ----------------
# #

if __name__ == '__main__':
    app.run_server(debug=True)
