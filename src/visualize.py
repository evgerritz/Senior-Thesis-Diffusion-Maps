import matplotlib as plt
import pandas as pd
import pickle

# special/interactive plotting
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from dash import Dash, dcc, html, Input, Output, no_update, jupyter_dash
import plotly.graph_objects as go
import flask
from mpl_toolkits.mplot3d import Axes3D

def plot_3d():
    # not working
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dmaps[0][:,0], dmaps[0][:,1], dmaps[0][:,2], c=y_subset)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(dmaps200[0][:,0], dmaps200[0][:,1], dmaps200[0][:,2], c=ysub200)
    plt.show()


def plot_images_as_points(dmap, coord_f1, coord_f2):
    fig, ax = plt.subplots(figsize=(7,7))

    for pt_x, pt_y, img in zip(dmap[:,coord_f1], dmap[:,coord_f2], X_sub_lin.reshape(len(X_sub_lin),64,64)):
        ab = AnnotationBbox(OffsetImage(img, zoom=0.4, cmap='gray', alpha=0.7), (pt_x, pt_y), frameon=False)
        ax.add_artist(ab)

    ax.scatter(dmap[:,coord_f1], dmap[:,coord_f2], s=1)#, c=y_subset, s=200)
    ax.set_xticks([])
    ax.set_yticks([])

def dash_visualizer(dmap, X, y, num_samples, calligraphers):
    df = pd.DataFrame()
    df['mat'] = pd.Series([x for x in X])
    df['coord0'] = dmap[:,0].real
    df['coord1'] = dmap[:,1].real
    df['coord2'] = dmap[:,2].real
    df['class_no'] = y
    for i in range(len(subset)):
        df.loc[i*num_samples:(i+1)*num_samples, 'Calligrapher'] = calligraphers[i]

    imgs_dir = 'assets/200_imgs/'
    file_strs = []
    for i in range(len(df)):
        #plt.imshow(df['img'][i],cmap='gray')
        #plt.xticks([])
        #plt.yticks([])
        file_str = imgs_dir+str(i)+'.png'
        #plt.savefig(file_str, bbox_inches='tight', pad_inches=0)
        file_strs.append(file_str)

    df['img'] = file_strs

    df['Calligrapher_zh'] = X.full_label(df['Calligrapher'])
    df['Calligrapher_en'] = X.full_label(df['Calligrapher'], chinese=False)


#jupyter_dash.infer_jupyter_proxy_config()
    fig = go.Figure(data=[
        go.Scatter(
            x=df["coord0"],
            y=df["coord1"],
            mode="markers",
            marker=dict(
                colorscale='viridis',
                color=df["class_no"],
                line={"color": "#444"},
                reversescale=True,
                sizeref=45,
                sizemode="diameter",
                opacity=0.8,
            )
        )
    ])

# turn off native plotly.js hover effects - make sure to use
# hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        xaxis=dict(title='coord0'),
        yaxis=dict(title='coord1'),
        plot_bgcolor='rgba(255,255,255,0.1)'
    )

    app = Dash('dash test')

    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True, style={'width': '95vh', 'height': '95vh', 'margin':'0 auto'}),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-basic-2", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]

        df_row = df.iloc[num]
        #img_src = 'file:///home/evgerritz/Dropbox/Yale/CPSC490/code/' + df_row['img']
        img_src = df_row['img']
        name_en = df_row['Calligrapher_en']
        name_zh = df_row['Calligrapher_zh']

        children = [
            html.Div([
                html.Img(src=img_src, style={"width": "100%"}),
                html.Br(),
                #html.P(children = [
                    html.Span(
                        f"{name_zh}", 
                        style={"color": "darkblue", "font-size" : "1em"}
                    ), 
                    html.Br(),
                    html.Span(
                        f"({name_en})", 
                        style={"color": "darkblue", "font-size" : "0.5em"}
                    )
                #]),
                
            ], style={'width': '75px', 'white-space': 'nowrap'})
        ]

        return True, bbox, children

    return app

#dash_visualizer().run()