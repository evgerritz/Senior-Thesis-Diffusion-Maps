import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# special/interactive plotting
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from dash import Dash, dcc, html, Input, Output, no_update, jupyter_dash
import plotly.graph_objects as go
import flask
from mpl_toolkits.mplot3d import Axes3D

def plot_dmap(embedding, coords=[(0,1)]):
    ncols = len(coords)
    fig, axes = plt.subplots(1,ncols,figsize=(ncols*4,1*4))
    #fig.suptitle(title)
    fig.subplots_adjust(hspace=.6) #adjust vertical spacing between subplots
    dmap = embedding.dmap
    for col in range(ncols):
        ax = axes[col]
        coord0, coord1 = coords[col]
        ax.scatter(dmap[:,coord0], dmap[:,coord1], c=embedding.y_sub)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'{coord0=}')
        ax.set_ylabel(f'{coord1=}')
    fig.tight_layout()

def plot_3d(embedding):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    dmap = embedding.dmap
    ax.scatter(dmap[:,0], dmap[:,1], dmap[:,2], c=embedding.y_sub)
    plt.show()

def plot_images_as_points(embedding, title, fname, coord_f1=0, coord_f2=1):
    fig, ax = plt.subplots(figsize=(6,6))
    dmap = embedding.dmap
    for pt_x, pt_y, img in zip(dmap[:,coord_f1], dmap[:,coord_f2], embedding.X_sub_lin.reshape(-1,64,64)):
        ab = AnnotationBbox(OffsetImage(img, zoom=0.4, cmap='gray', alpha=0.7), (pt_x, pt_y), frameon=False)
        ax.add_artist(ab)

    ax.set_title(title, fontsize=15)
    #ax.scatter(dmap[:,coord_f1], dmap[:,coord_f2], s=1)#, c=y_subset, s=200)
    ax.scatter(dmap[:,coord_f1], dmap[:,coord_f2], c=embedding.y_sub, s=30)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    plt.savefig('../../res/{fname}.png')

def plot_cluster_images(embedding, title, fname, coord_f1=0, coord_f2=1):
    fig, ax = plt.subplots(figsize=(8,8), dpi=200)
    dmap = embedding.dmap
    mean_xs = []
    mean_ys = []
    min_x = np.inf; max_y = -np.inf
    full_label = lambda y, zh: embedding.data.full_label(embedding.data.get_label(class_no=y), chinese=zh)
    for y_val in np.unique(embedding.y_sub):
        where = (embedding.y_sub == y_val).reshape(-1)
        xs = dmap[where, coord_f1]
        ys = dmap[where, coord_f2]
        min_x = min((min_x, min(xs)))
        max_y = max((max_y, max(ys)))
        imgs = embedding.X_sub_lin.reshape(-1,64,64)
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        tol = 7e-6

        mean_xs.append(mean_x)
        mean_ys.append(mean_y)

        ax.scatter(xs,ys, cmap='tab20', alpha=0.75, s=30, label=f'{full_label(y_val, False)} ({full_label(y_val, True)})')
        dists = ((xs-mean_x)**2 + (ys-mean_y)**2)
        closest = np.argmin(dists)
        img = imgs[where][closest]
        x_jitter = np.random.random()*np.mean(dmap[:,coord_f1])*50
        y_jitter = np.random.random()*np.mean(dmap[:,coord_f2])*50
        ab = AnnotationBbox(OffsetImage(img, zoom=0.4, cmap='gray', alpha=0.9), (xs[closest]+x_jitter, ys[closest]+y_jitter), frameon=False)
        ax.add_artist(ab)

    pointwise_nmi = embedding.nmi_labs(True)
    ax.text(min_x, max_y, f'NMI: {round(pointwise_nmi,3)}', ha='left', va='top', fontsize=11)

    ax.set_title(title, fontsize=15)
    ax.legend(loc='lower right', fontsize=7)
    #ax.scatter(dmap[:,coord_f1], dmap[:,coord_f2], s=1)#, c=y_subset, s=200)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.savefig(f'../../res/{fname}.png')
    plt.show()

def dash_visualizer(embedding, refresh_data=True):
    df = pd.DataFrame()
    X = embedding.X_sub_lin
    dmap = embedding.dmap
    df['mat'] = pd.Series([x.reshape(64, 64) for x in X])
    df['coord0'] = dmap[:,0].real
    df['coord1'] = dmap[:,1].real
    df['coord2'] = dmap[:,2].real
    df['class_no'] = embedding.y_sub
    subset = embedding.subset
    num_samples = embedding.num_samples
    for i in range(len(subset)):
        df.loc[i*num_samples:(i+1)*num_samples, 'Calligrapher'] = subset[i]

    imgs_dir = '../assets/dash_images/'
    file_strs = []
    if refresh_data:
        for i in tqdm(range(len(df))):
            file_str = imgs_dir+str(i)+'.png'
            if os.path.exists(file_str): continue
            plt.imshow(df['mat'][i],cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(file_str, bbox_inches='tight', pad_inches=0, dpi=50)
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
