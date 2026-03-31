import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =========================================================
# TASK 1 - FINAL PRESENTATION VERSION
# Only four libraries are used:
# numpy, pandas, networkx, matplotlib
# =========================================================

# ---------------------------------------------------------
# 1. User chooses which edge attribute to display
# ---------------------------------------------------------
print("Singapore MRT Schematic Network")
print("Choose the distance unit to display on the graph:")
print("1 - Kilometres")
print("2 - Miles")

choice = input("Enter 1 or 2: ").strip()
if choice == "2":
    edge_attribute = "miles"
    unit_text = "mi"
else:
    edge_attribute = "km"
    unit_text = "km"

# ---------------------------------------------------------
# 2. Create graph
# ---------------------------------------------------------
G = nx.Graph()

# ---------------------------------------------------------
# 3. Line definitions
# ---------------------------------------------------------
LINE_COLOURS = {
    "East West Line": "#1ea64b",
    "Circle Line": "#f39c12",
    "Downtown Line": "#1f5fbf",
    "Downtown Line Extension": "#1f5fbf",
    "North East Line": "#9b59b6",
    "Thomson-East Coast Line": "#8c5a2b"
}

line_info = {
    "EWL": {"name": "East West Line", "color": LINE_COLOURS["East West Line"], "style": "solid", "width": 4.8, "alpha": 1.0},
    "CCL": {"name": "Circle Line", "color": LINE_COLOURS["Circle Line"], "style": "solid", "width": 4.8, "alpha": 1.0},
    "DTL": {"name": "Downtown Line", "color": LINE_COLOURS["Downtown Line"], "style": "solid", "width": 4.8, "alpha": 1.0},
    "NEL": {"name": "North East Line", "color": LINE_COLOURS["North East Line"], "style": "solid", "width": 4.8, "alpha": 1.0},
    "TEL": {"name": "Thomson-East Coast Line", "color": LINE_COLOURS["Thomson-East Coast Line"], "style": "dashed", "width": 3.7, "alpha": 1.0},
    "DTL_EXT": {"name": "Downtown Line Extension", "color": LINE_COLOURS["Downtown Line Extension"], "style": "dashed", "width": 3.1, "alpha": 0.95}
}

# ---------------------------------------------------------
# 4. Station lists
# ---------------------------------------------------------
ewl_stations = [
    "Aljunied", "Paya Lebar", "Eunos", "Kembangan",
    "Bedok", "Tanah Merah", "Simei", "Tampines", "Pasir Ris"
]

ewl_branch_stations = [
    "Tanah Merah", "Expo", "Changi Airport"
]

ccl_stations = [
    "Serangoon", "Bartley", "Tai Seng", "MacPherson",
    "Paya Lebar", "Dakota", "Mountbatten", "Stadium",
    "Nicoll Highway", "Promenade"
]

dtl_stations = [
    "Little India", "Jalan Besar", "Bendemeer", "Geylang Bahru",
    "Mattar", "MacPherson", "Ubi", "Kaki Bukit",
    "Bedok North", "Bedok Reservoir", "Tampines West",
    "Tampines", "Tampines East", "Upper Changi", "Expo"
]

dtl_extension_stations = [
    "Expo", "Xilin", "Sungei Bedok"
]

nel_stations = [
    "Little India", "Farrer Park", "Boon Keng",
    "Potong Pasir", "Woodleigh", "Serangoon", "Kovan"
]

tel_stations = [
    "Tanjong Katong", "Marine Parade", "Marine Terrace",
    "Siglap", "Bayshore", "Bedok South", "Sungei Bedok"
]

# ---------------------------------------------------------
# 5. Manual schematic positions
# ---------------------------------------------------------
pos = {
    # coursework core
    "Mattar": (0.0, 0.0),
    "MacPherson": (2.2, 2.0),
    "Ubi": (4.4, 4.0),
    "Kaki Bukit": (6.6, 6.0),

    "Aljunied": (-3.0, -2.0),
    "Paya Lebar": (2.2, -2.0),
    "Eunos": (4.4, 0.0),
    "Kembangan": (6.6, 2.0),

    "Tai Seng": (0.0, 4.0),
    "Dakota": (2.2, -4.4),

    # dtl west
    "Little India": (-11.8, -2.0),
    "Jalan Besar": (-9.4, -2.0),
    "Bendemeer": (-7.0, -2.0),
    "Geylang Bahru": (-3.8, -0.9),

    # dtl east
    "Bedok North": (9.4, 6.0),
    "Bedok Reservoir": (12.2, 6.0),
    "Tampines West": (15.0, 6.0),
    "Tampines": (18.0, 4.0),
    "Tampines East": (18.0, 1.6),
    "Upper Changi": (18.0, -0.8),
    "Expo": (18.0, -3.2),

    # extension
    "Xilin": (17.2, -5.8),
    "Sungei Bedok": (16.0, -8.9),

    # ewl east
    "Bedok": (9.4, 2.0),
    "Tanah Merah": (12.2, 2.0),
    "Simei": (15.0, 2.0),
    "Pasir Ris": (18.0, 7.8),
    "Changi Airport": (21.0, -3.2),

    # ccl
    "Bartley": (-1.8, 5.4),
    "Serangoon": (-4.0, 7.2),
    "Mountbatten": (3.3, -6.3),
    "Stadium": (4.6, -7.9),
    "Nicoll Highway": (6.1, -9.5),
    "Promenade": (8.0, -10.9),

    # nel
    "Farrer Park": (-9.9, -0.1),
    "Boon Keng": (-7.8, 1.9),
    "Potong Pasir": (-5.8, 3.9),
    "Woodleigh": (-4.7, 5.8),
    "Kovan": (-1.8, 8.7),

    # tel
    "Tanjong Katong": (8.7, -13.8),
    "Marine Parade": (11.2, -12.4),
    "Marine Terrace": (13.7, -11.1),
    "Siglap": (15.4, -9.6),
    "Bayshore": (16.8, -8.0),
    "Bedok South": (18.3, -6.6)
}

# ---------------------------------------------------------
# 6. Temporary distances
# ---------------------------------------------------------
distance_km = {
    ("Aljunied", "Paya Lebar"): 1.7,
    ("Paya Lebar", "Eunos"): 1.6,
    ("Eunos", "Kembangan"): 1.4,
    ("Kembangan", "Bedok"): 1.8,
    ("Bedok", "Tanah Merah"): 2.0,
    ("Tanah Merah", "Simei"): 1.7,
    ("Simei", "Tampines"): 1.8,
    ("Tampines", "Pasir Ris"): 2.1,

    ("Tanah Merah", "Expo"): 1.9,
    ("Expo", "Changi Airport"): 2.2,

    ("Serangoon", "Bartley"): 1.3,
    ("Bartley", "Tai Seng"): 1.4,
    ("Tai Seng", "MacPherson"): 1.1,
    ("MacPherson", "Paya Lebar"): 1.5,
    ("Paya Lebar", "Dakota"): 1.3,
    ("Dakota", "Mountbatten"): 1.2,
    ("Mountbatten", "Stadium"): 1.2,
    ("Stadium", "Nicoll Highway"): 1.4,
    ("Nicoll Highway", "Promenade"): 1.2,

    ("Little India", "Jalan Besar"): 1.0,
    ("Jalan Besar", "Bendemeer"): 1.1,
    ("Bendemeer", "Geylang Bahru"): 1.3,
    ("Geylang Bahru", "Mattar"): 1.3,
    ("Mattar", "MacPherson"): 1.0,
    ("MacPherson", "Ubi"): 1.2,
    ("Ubi", "Kaki Bukit"): 1.3,
    ("Kaki Bukit", "Bedok North"): 1.4,
    ("Bedok North", "Bedok Reservoir"): 1.3,
    ("Bedok Reservoir", "Tampines West"): 1.5,
    ("Tampines West", "Tampines"): 1.2,
    ("Tampines", "Tampines East"): 1.1,
    ("Tampines East", "Upper Changi"): 1.2,
    ("Upper Changi", "Expo"): 1.1,

    ("Expo", "Xilin"): 1.3,
    ("Xilin", "Sungei Bedok"): 1.4,

    ("Little India", "Farrer Park"): 1.0,
    ("Farrer Park", "Boon Keng"): 1.2,
    ("Boon Keng", "Potong Pasir"): 1.5,
    ("Potong Pasir", "Woodleigh"): 1.3,
    ("Woodleigh", "Serangoon"): 1.1,
    ("Serangoon", "Kovan"): 1.4,

    ("Tanjong Katong", "Marine Parade"): 1.2,
    ("Marine Parade", "Marine Terrace"): 1.0,
    ("Marine Terrace", "Siglap"): 1.1,
    ("Siglap", "Bayshore"): 1.0,
    ("Bayshore", "Bedok South"): 0.9,
    ("Bedok South", "Sungei Bedok"): 1.1
}

# ---------------------------------------------------------
# 7. Add stations and edges
# ---------------------------------------------------------
def add_line_to_graph(graph, station_list, line_code):
    for station in station_list:
        if station not in graph.nodes:
            graph.add_node(station, lines=[line_code])
        else:
            if "lines" not in graph.nodes[station]:
                graph.nodes[station]["lines"] = [line_code]
            elif line_code not in graph.nodes[station]["lines"]:
                graph.nodes[station]["lines"].append(line_code)

    for i in range(len(station_list) - 1):
        u = station_list[i]
        v = station_list[i + 1]
        km_value = distance_km[(u, v)]
        miles_value = np.round(km_value * 0.621371, 2)

        if graph.has_edge(u, v):
            if line_code not in graph[u][v]["lines"]:
                graph[u][v]["lines"].append(line_code)
        else:
            graph.add_edge(
                u, v,
                km=km_value,
                miles=miles_value,
                lines=[line_code],
                line=line_info[line_code]["name"],
                dashed=(line_info[line_code]["style"] == "dashed")
            )

add_line_to_graph(G, ewl_stations, "EWL")
add_line_to_graph(G, ewl_branch_stations, "EWL")
add_line_to_graph(G, ccl_stations, "CCL")
add_line_to_graph(G, dtl_stations, "DTL")
add_line_to_graph(G, dtl_extension_stations, "DTL_EXT")
add_line_to_graph(G, nel_stations, "NEL")
add_line_to_graph(G, tel_stations, "TEL")

assert nx.is_connected(G), "Graph is not connected."

# ---------------------------------------------------------
# 8. Figure layout
# Main map enlarged to use screen better
# ---------------------------------------------------------
fig = plt.figure(figsize=(22, 14), facecolor="#eeeeee")

ax = fig.add_axes([0.04, 0.07, 0.78, 0.86])
ax.set_facecolor("#eeeeee")

key_ax = fig.add_axes([0.81, 0.09, 0.15, 0.17])
key_ax.set_facecolor("white")
for spine in key_ax.spines.values():
    spine.set_edgecolor("#444444")
    spine.set_linewidth(1.0)
key_ax.set_xticks([])
key_ax.set_yticks([])
key_ax.set_xlim(0, 1)
key_ax.set_ylim(0, 1)

# ---------------------------------------------------------
# 9. Title
# ---------------------------------------------------------
fig.text(0.04, 0.965, "Singapore MRT Schematic Network",
         fontsize=15, fontweight="bold", color="#222222", ha="left", va="top")
fig.text(0.04, 0.944, "Expanded from the original coursework diagram",
         fontsize=9.5, color="#555555", ha="left", va="top")
fig.text(0.04, 0.928, "Displayed edge attribute: " + unit_text,
         fontsize=9.5, color="#555555", ha="left", va="top")

# ---------------------------------------------------------
# 10. Draw edges
# ---------------------------------------------------------
for line_code, info in line_info.items():
    line_name = info["name"]

    solid_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d["line"] == line_name and not d.get("dashed")
    ]
    dashed_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d["line"] == line_name and d.get("dashed")
    ]

    if solid_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=solid_edges,
            edge_color=info["color"], width=info["width"],
            style="solid", alpha=1.0, ax=ax
        )

    if dashed_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=dashed_edges,
            edge_color=info["color"], width=info["width"],
            style="dashed", alpha=info["alpha"], ax=ax
        )

# ---------------------------------------------------------
# 11. Edge labels
# Smaller so overview remains readable
# ---------------------------------------------------------
edge_labels = {(u, v): f"{d[edge_attribute]}{unit_text}" for u, v, d in G.edges(data=True)}

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=4.0,
    font_color="#555555",
    bbox=dict(boxstyle="round,pad=0.02", fc="#eeeeee", ec="none", alpha=0.35),
    rotate=False,
    ax=ax
)

# ---------------------------------------------------------
# 12. Nodes
# ---------------------------------------------------------
node_lines = {n: set() for n in G.nodes()}
for u, v, d in G.edges(data=True):
    node_lines[u].add(d["line"])
    node_lines[v].add(d["line"])

interchange_nodes = [n for n, ls in node_lines.items() if len(ls) > 1]
single_nodes = [n for n, ls in node_lines.items() if len(ls) == 1]

nx.draw_networkx_nodes(
    G, pos,
    nodelist=interchange_nodes,
    node_color="#f3f3f3",
    node_size=400,
    edgecolors="#9a9a9a",
    linewidths=1.8,
    ax=ax
)

for node in single_nodes:
    line_name = list(node_lines[node])[0]
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[node],
        node_color=LINE_COLOURS[line_name],
        node_size=240,
        edgecolors="white",
        linewidths=1.6,
        ax=ax
    )

# ---------------------------------------------------------
# 13. Station labels
# ---------------------------------------------------------
label_pos = {
    "Mattar": (0.0, -0.68),
    "MacPherson": (2.4, 2.78),
    "Ubi": (4.4, 4.98),
    "Kaki Bukit": (6.6, 7.20),

    "Aljunied": (-4.15, -2.22),
    "Paya Lebar": (2.4, -2.86),
    "Eunos": (4.4, 0.48),
    "Kembangan": (6.6, 2.72),

    "Tai Seng": (-1.0, 5.00),
    "Dakota": (2.4, -5.10),

    "Little India": (-13.3, -2.22),
    "Jalan Besar": (-10.1, -2.88),
    "Bendemeer": (-7.2, -2.88),
    "Geylang Bahru": (-5.0, -1.26),

    "Bedok": (10.2, 2.72),
    "Tanah Merah": (12.2, 2.72),
    "Simei": (15.0, 2.72),
    "Tampines": (19.4, 4.98),
    "Pasir Ris": (19.4, 8.76),

    "Bedok North": (9.4, 7.20),
    "Bedok Reservoir": (12.2, 7.20),
    "Tampines West": (15.0, 7.20),
    "Tampines East": (19.4, 2.00),
    "Upper Changi": (19.4, -0.48),
    "Expo": (19.4, -4.05),
    "Changi Airport": (22.95, -3.25),

    "Xilin": (18.55, -5.32),
    "Sungei Bedok": (14.6, -10.02),
    "Bedok South": (19.55, -6.60),

    "Bartley": (-2.7, 6.08),
    "Serangoon": (-5.2, 7.90),
    "Mountbatten": (3.5, -6.70),
    "Stadium": (4.9, -8.40),
    "Nicoll Highway": (6.6, -10.10),
    "Promenade": (8.8, -11.80),

    "Farrer Park": (-10.95, 0.28),
    "Boon Keng": (-8.55, 2.28),
    "Potong Pasir": (-6.35, 4.30),
    "Woodleigh": (-5.15, 6.36),
    "Kovan": (-2.55, 9.68),

    "Tanjong Katong": (8.7, -14.05),
    "Marine Parade": (11.2, -12.78),
    "Marine Terrace": (14.4, -11.48),
    "Siglap": (16.1, -9.86),
    "Bayshore": (17.75, -8.00)
}

nx.draw_networkx_labels(
    G, label_pos,
    font_size=7.0,
    font_color="#111111",
    font_weight="normal",
    ax=ax
)

# ---------------------------------------------------------
# 14. Key
# ---------------------------------------------------------
key_ax.text(0.08, 0.92, "Key", fontsize=10.0, fontweight="bold",
            color="#222222", ha="left", va="top")

legend_items = [
    ("East West Line", LINE_COLOURS["East West Line"], "solid"),
    ("Circle Line", LINE_COLOURS["Circle Line"], "solid"),
    ("Downtown Line", LINE_COLOURS["Downtown Line"], "solid"),
    ("North East Line", LINE_COLOURS["North East Line"], "solid"),
    ("Thomson-East Coast Line", LINE_COLOURS["Thomson-East Coast Line"], "dashed"),
    ("Downtown Line Extension", LINE_COLOURS["Downtown Line Extension"], "dashed")
]

for i in range(len(legend_items)):
    name, color, style = legend_items[i]
    y = 0.80 - i * 0.12
    key_ax.plot([0.08, 0.38], [y, y], color=color, linewidth=2.8, linestyle=style)
    key_ax.text(0.43, y, name, fontsize=7.9, color="#222222", va="center", ha="left")

key_ax.text(0.08, 0.06, "Dashed lines indicate future line/extension",
            fontsize=6.9, color="#555555", ha="left")

# ---------------------------------------------------------
# 15. Final formatting
# Tighter limits so the network fills the screen better
# ---------------------------------------------------------
ax.set_xlim(-13.8, 23.6)
ax.set_ylim(-15.2, 10.2)
ax.set_aspect("equal")
ax.axis("off")

plt.show()