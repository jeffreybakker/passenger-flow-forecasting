from collections import defaultdict
from datetime import datetime
from itertools import tee
from typing import Tuple, Dict, Callable, Optional, Union, Iterable

import numpy as np
from PIL import Image
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from networkx import shortest_path


def read_stations() -> pd.DataFrame:
    try:
        res = pd.read_csv('data/stations.csv')
    except FileNotFoundError:
        res = pd.read_csv('../data/stations.csv')

    return res.assign(
        coords=[
            tuple(map(lambda value: int(value.strip()), coords.strip('()').split(',')))
            for coords in res.coords],
        neighbours=[
            list(map(lambda n: n.strip('\' '), neighbours.strip('[]').split(',')))
            for neighbours in res.neighbours])


def read_bart(year: int) -> pd.DataFrame:
    try:
        res = pd.read_csv(
            f'data/bart-od-{year}.csv.gz', compression='gzip',
            names=['date', 'hour', 'origin', 'destination', 'passengers']
        )
    except FileNotFoundError:
        res = pd.read_csv(
            f'../data/bart-od-{year}.csv.gz', compression='gzip',
            names=['date', 'hour', 'origin', 'destination', 'passengers']
        )

    transform = {row['abbreviations']: row['codes'] for _, row in stations.iterrows()}
    res = res.assign(
        origin=[transform[origin] for origin in res.origin],
        destination=[transform[destination] for destination in res.destination]
    )

    res['datetime'] = res.apply(
        lambda row: datetime.strptime(f'{row.date} {row.hour:02d}:00:00', '%Y-%m-%d %H:%M:%S'),
        axis=1)

    res.drop(columns=['date', 'hour'], inplace=True)

    return res


def construct_graph(stations: pd.DataFrame) -> Tuple[nx.DiGraph, Dict[str, Tuple[int, int]]]:
    g = nx.DiGraph()

    stations_map = {}
    for index, row in stations.iterrows():
        g.add_node(row['codes'])
        stations_map[row['codes']] = row['coords']

        for neighbour in row['neighbours']:
            g.add_edge(row['codes'], neighbour)

    return g, stations_map


def pairwise(iterable: Iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


# Compute the shortest paths
stations = read_stations()
g, stations_map = construct_graph(stations)
paths = defaultdict(dict)
for start in stations.codes:
    for dest in stations.codes:
        paths[start][dest] = shortest_path(g, start, dest)


def compute_flow(data: pd.DataFrame) -> pd.DataFrame:
    res = data \
        .assign(path=[list(pairwise(paths[row['origin']][row['destination']])) for _, row in data.iterrows()]) \
        .explode('path').dropna()

    return res \
        .assign(
            origin=[path[0] for path in res.path],
            destination=[path[1] for path in res.path]
        )[['origin', 'destination', 'passengers']] \
        .groupby(['origin', 'destination']).sum() \
        .reset_index()


def populate_flow(g: nx.DiGraph, flow: pd.DataFrame, ts_key: Optional[str] = None) -> nx.DiGraph:
    for _, row in flow.iterrows():
        a, b, amt = row['origin'], row['destination'], row['passengers']
        if 'flow' in g[a][b]:
            g[a][b]['flow'] += amt
        else:
            g[a][b]['flow'] = amt

        if ts_key is not None and 'ts' not in g[a][b]:
                g[a][b]['ts'] = {}

        if ts_key is not None:
            if ts_key in g[a][b]['ts']:
                g[a][b]['ts'][ts_key] += amt
            else:
                g[a][b]['ts'][ts_key] = amt

    return g


def draw_graph(g: nx.DiGraph,
               pos: Dict[str, Tuple[int, int]],
               ax: Optional[plt.axis] = None,
               ts_key: Optional[str] = None,
               overlay: float = 0.25,
               nodes: Union[int, bool] = 75,
               scale: float = 1.0,
               cmap=plt.cm.plasma,
               vmin: float = 0.0, vmax: float = 1000.0):
    if not ax:
        plt.figure(figsize=(7, 6))
        plt.grid(False)
    else:
        ax.grid(False)

    if overlay > 0.0:
        # noinspection PyTypeChecker
        img = np.array(Image.open('./assets/bart-map.png'))
        img[:, :, 3] = img[:, :, 3] * overlay

        (ax if ax is not None else plt).imshow(img)

    if nodes:
        nx.draw_networkx_nodes(g, pos=pos, ax=ax,
                               node_color='#000000',
                               node_size=5 * scale)

    edge_value: Callable[[str, str], int] = lambda a, b: \
        g[a][b]['ts'][ts_key] \
        if ts_key is not None else \
        (g[a][b]['flow'] if 'flow' in g[a][b] else 0)
    nx.draw_networkx_edges(g, pos=pos, ax=ax,
                           node_size=5 * scale if nodes else 0,
                           edge_color=[edge_value(a, b) for a, b in g.edges()],
                           edge_cmap=cmap,
                           edge_vmin=vmin, edge_vmax=vmax,
                           alpha=0.7,
                           arrowsize=5 * scale,
                           width=min(2, int(2 * scale)),
                           min_source_margin=5 * scale)
