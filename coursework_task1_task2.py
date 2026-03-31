import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =========================================================
# Coursework - Task 1 and Task 2
# Bird-flight (great-circle) distances between stations
# using station latitude/longitude coordinates.
# Only the allowed four libraries are imported.
# =========================================================

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


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0088
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(earth_radius_km * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def build_edge_dataframe(station_df):
    station_lookup = station_df.set_index("station")[["latitude", "longitude"]].to_dict("index")

    lines = {
        "East West Line": {
            "stations": [
                "Aljunied", "Paya Lebar", "Eunos", "Kembangan",
                "Bedok", "Tanah Merah", "Simei", "Tampines", "Pasir Ris"
            ],
            "future": 0,
            "color": "#1ea64b",
            "style": "solid"
        },
        "East West Line Branch": {
            "stations": ["Tanah Merah", "Expo", "Changi Airport"],
            "future": 0,
            "color": "#1ea64b",
            "style": "solid"
        },
        "Circle Line": {
            "stations": [
                "Serangoon", "Bartley", "Tai Seng", "MacPherson",
                "Paya Lebar", "Dakota", "Mountbatten", "Stadium",
                "Nicoll Highway", "Promenade"
            ],
            "future": 0,
            "color": "#f39c12",
            "style": "solid"
        },
        "Downtown Line": {
            "stations": [
                "Little India", "Jalan Besar", "Bendemeer", "Geylang Bahru",
                "Mattar", "MacPherson", "Ubi", "Kaki Bukit",
                "Bedok North", "Bedok Reservoir", "Tampines West",
                "Tampines", "Tampines East", "Upper Changi", "Expo"
            ],
            "future": 0,
            "color": "#1f5fbf",
            "style": "solid"
        },
        "Downtown Line Extension": {
            "stations": ["Expo", "Xilin", "Sungei Bedok"],
            "future": 1,
            "color": "#1f5fbf",
            "style": "dashed"
        },
        "North East Line": {
            "stations": [
                "Little India", "Farrer Park", "Boon Keng",
                "Potong Pasir", "Woodleigh", "Serangoon", "Kovan"
            ],
            "future": 0,
            "color": "#9b59b6",
            "style": "solid"
        },
        "Thomson-East Coast Line": {
            "stations": [
                "Marine Parade", "Marine Terrace", "Siglap",
                "Bayshore", "Bedok South", "Sungei Bedok"
            ],
            "future": 1,
            "color": "#8c5a2b",
            "style": "dashed"
        }
    }

    rows = []
    for line_name, info in lines.items():
        stations = info["stations"]
        for station_1, station_2 in zip(stations, stations[1:]):
            lat1 = station_lookup[station_1]["latitude"]
            lon1 = station_lookup[station_1]["longitude"]
            lat2 = station_lookup[station_2]["latitude"]
            lon2 = station_lookup[station_2]["longitude"]
            km = round(haversine_km(lat1, lon1, lat2, lon2), 3)
            miles = round(km * 0.621371, 3)
            rows.append({
                "station_1": station_1,
                "station_2": station_2,
                "line": line_name,
                "future": info["future"],
                "color": info["color"],
                "style": info["style"],
                "km": km,
                "miles": miles
            })

    return pd.DataFrame(rows)


def build_graph(edge_df):
    graph = nx.Graph()
    for _, row in edge_df.iterrows():
        graph.add_edge(
            row["station_1"],
            row["station_2"],
            line=row["line"],
            future=int(row["future"]),
            color=row["color"],
            style=row["style"],
            km=float(row["km"]),
            miles=float(row["miles"])
        )
    return graph


def build_positions():
    return {
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
        "Little India": (-11.8, -2.0),
        "Jalan Besar": (-9.4, -2.0),
        "Bendemeer": (-7.0, -2.0),
        "Geylang Bahru": (-3.8, -0.9),
        "Bedok North": (9.4, 6.0),
        "Bedok Reservoir": (12.2, 6.0),
        "Tampines West": (15.0, 6.0),
        "Tampines": (18.0, 4.0),
        "Tampines East": (18.0, 1.6),
        "Upper Changi": (18.0, -0.8),
        "Expo": (18.0, -3.2),
        "Xilin": (17.2, -5.8),
        "Sungei Bedok": (16.0, -8.9),
        "Bedok": (9.4, 2.0),
        "Tanah Merah": (12.2, 2.0),
        "Simei": (15.0, 2.0),
        "Pasir Ris": (18.0, 7.8),
        "Changi Airport": (21.0, -3.2),
        "Bartley": (-1.8, 5.4),
        "Serangoon": (-4.0, 7.2),
        "Mountbatten": (3.3, -6.3),
        "Stadium": (4.6, -7.9),
        "Nicoll Highway": (6.1, -9.5),
        "Promenade": (8.0, -10.9),
        "Farrer Park": (-9.9, -0.1),
        "Boon Keng": (-7.8, 1.9),
        "Potong Pasir": (-5.8, 3.9),
        "Woodleigh": (-4.7, 5.8),
        "Kovan": (-1.8, 8.7),
        "Marine Parade": (11.2, -12.4),
        "Marine Terrace": (13.7, -11.1),
        "Siglap": (15.4, -9.6),
        "Bayshore": (16.8, -8.0),
        "Bedok South": (18.3, -6.6)
    }


def build_label_positions():
    return {
        "Mattar": (0.0, -0.68),
        "MacPherson": (2.2, 2.78),
        "Ubi": (4.4, 4.98),
        "Kaki Bukit": (6.6, 7.20),
        "Aljunied": (-4.15, -2.22),
        "Paya Lebar": (2.2, -2.86),
        "Eunos": (4.4, 0.48),
        "Kembangan": (6.6, 2.72),
        "Tai Seng": (-1.0, 5.00),
        "Dakota": (2.2, -5.10),
        "Little India": (-13.3, -2.24),
        "Jalan Besar": (-10.1, -2.88),
        "Bendemeer": (-7.2, -2.88),
        "Geylang Bahru": (-5.0, -1.26),
        "Bedok": (9.4, 2.72),
        "Tanah Merah": (12.2, 2.72),
        "Simei": (15.0, 2.72),
        "Tampines": (18.0, 4.98),
        "Pasir Ris": (18.0, 8.76),
        "Bedok North": (9.4, 7.20),
        "Bedok Reservoir": (12.2, 7.20),
        "Tampines West": (15.0, 7.20),
        "Tampines East": (18.0, 2.00),
        "Upper Changi": (18.0, -0.48),
        "Expo": (18.0, -4.05),
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
        "Marine Parade": (11.2, -12.78),
        "Marine Terrace": (14.4, -11.48),
        "Siglap": (16.1, -9.86),
        "Bayshore": (17.75, -8.00)
    }


def draw_network(graph, edge_df, distance_attr, output_path):
    pos = build_positions()
    label_pos = build_label_positions()

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

    fig.text(0.04, 0.965, "Singapore MRT Schematic Network", fontsize=15,
             fontweight="bold", color="#222222", ha="left", va="top")
    fig.text(0.04, 0.944, "Task 1 - bird-flight distances from station coordinates", fontsize=9.5,
             color="#555555", ha="left", va="top")
    fig.text(0.04, 0.928, f"Displayed edge attribute: {distance_attr}", fontsize=9.5,
             color="#555555", ha="left", va="top")

    unique_lines = edge_df[["line", "color", "style"]].drop_duplicates()
    for _, line_row in unique_lines.iterrows():
        line_name = line_row["line"]
        edgelist = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d["line"] == line_name
        ]
        width = 4.8 if line_row["style"] == "solid" else 3.3
        nx.draw_networkx_edges(
            graph, pos, edgelist=edgelist, edge_color=line_row["color"],
            width=width, style=line_row["style"], alpha=1.0, ax=ax
        )

    edge_labels = {}
    for u, v, d in graph.edges(data=True):
        display_value = round(d[distance_attr], 1)
        suffix = "km" if distance_attr == "km" else "mi"
        edge_labels[(u, v)] = f"{display_value}{suffix}"

    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels,
        font_size=4.0, font_color="#555555",
        bbox=dict(boxstyle="round,pad=0.02", fc="#eeeeee", ec="none", alpha=0.35),
        rotate=False, ax=ax
    )

    node_lines = {n: set() for n in graph.nodes()}
    for u, v, d in graph.edges(data=True):
        node_lines[u].add(d["line"])
        node_lines[v].add(d["line"])

    interchange_nodes = [n for n, lines in node_lines.items() if len(lines) > 1]
    single_nodes = [n for n, lines in node_lines.items() if len(lines) == 1]

    nx.draw_networkx_nodes(
        graph, pos, nodelist=interchange_nodes,
        node_color="#f3f3f3", node_size=400,
        edgecolors="#9a9a9a", linewidths=1.8, ax=ax
    )

    for node in single_nodes:
        line_name = list(node_lines[node])[0]
        line_color = unique_lines.loc[unique_lines["line"] == line_name, "color"].iloc[0]
        nx.draw_networkx_nodes(
            graph, pos, nodelist=[node],
            node_color=line_color, node_size=240,
            edgecolors="white", linewidths=1.6, ax=ax
        )

    nx.draw_networkx_labels(
        graph, label_pos,
        font_size=7.0, font_color="#111111",
        font_weight="normal", ax=ax
    )

    key_ax.text(0.08, 0.92, "Key", fontsize=10.0, fontweight="bold",
                color="#222222", ha="left", va="top")

    legend_items = [
        ("East West Line", "#1ea64b", "solid"),
        ("Circle Line", "#f39c12", "solid"),
        ("Downtown Line", "#1f5fbf", "solid"),
        ("North East Line", "#9b59b6", "solid"),
        ("Thomson-East Coast Line", "#8c5a2b", "dashed"),
        ("Downtown Line Extension", "#1f5fbf", "dashed")
    ]

    for i, (name, color, style) in enumerate(legend_items):
        y = 0.80 - i * 0.12
        key_ax.plot([0.08, 0.38], [y, y], color=color, linewidth=2.8, linestyle=style)
        key_ax.text(0.43, y, name, fontsize=7.9, color="#222222", va="center", ha="left")

    key_ax.text(0.08, 0.06, "Dashed lines indicate future line/extension",
                fontsize=6.9, color="#555555", ha="left")

    ax.set_xlim(-13.8, 23.6)
    ax.set_ylim(-15.2, 10.2)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def task2_statistics(edge_df):
    total_km = round(edge_df["km"].sum(), 3)
    total_miles = round(edge_df["miles"].sum(), 3)
    avg_km = round(edge_df["km"].mean(), 3)
    avg_miles = round(edge_df["miles"].mean(), 3)
    summary = pd.DataFrame({
        "metric": [
            "Total network length",
            "Average distance per edge"
        ],
        "km": [total_km, avg_km],
        "miles": [total_miles, avg_miles]
    })
    return summary


def main():
    coordinates_path = "station_coordinates.csv"
    station_df = pd.read_csv(coordinates_path)
    edge_df = build_edge_dataframe(station_df)
    graph = build_graph(edge_df)

    if not nx.is_connected(graph):
        raise ValueError("The generated graph is not connected.")

    output_image = "task1_map.png"
    draw_network(graph, edge_df, edge_attribute, output_image)

    edge_df.to_csv("edge_distances.csv", index=False)

    summary_df = task2_statistics(edge_df)
    summary_df.to_csv("task2_summary.csv", index=False)

    print("\nTASK 2 RESULTS")
    print(summary_df.to_string(index=False))
    print("\nFiles created:")
    print("- task1_map.png")
    print("- edge_distances.csv")
    print("- task2_summary.csv")


if __name__ == "__main__":
    main()
