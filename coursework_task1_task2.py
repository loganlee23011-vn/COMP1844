import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =====================================================================
# SINGAPORE MRT NETWORK – COMP1844 COURSEWORK
# Task 1: Visualise transport network graph with distance edge labels
# Task 2: Extract total and average network distances
# =====================================================================

# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
COLORS = {
    "East West Line":         "#1E9E63",
    "Changi Airport Line":    "#1E9E63",
    "North South Line":       "#D62828",
    "North East Line":        "#8E44AD",
    "Circle Line":            "#F39C12",
    "Downtown Line":          "#1D6FD6",
}
STYLES = {
    "East West Line":         "solid",
    "Changi Airport Line":    "solid",
    "North South Line":       "solid",
    "North East Line":        "solid",
    "Circle Line":            "solid",
    "Downtown Line":          "solid",
}
BACKGROUND = "#F4F4F4"

# ─────────────────────────────────────────────
# LINE DEFINITIONS
# ─────────────────────────────────────────────
MAP1_LINES = {
    "Circle Line": {
        "stations": ["Tai Seng", "MacPherson", "Paya Lebar", "Dakota"],
    },
    "Downtown Line": {
        "stations": ["Mattar", "MacPherson", "Ubi", "Kaki Bukit"],
    },
    "East West Line": {
        "stations": ["Aljunied", "Paya Lebar", "Eunos", "Kembangan"],
    },
}

MAP2_LINES = {
    "North South Line": {
        "stations": [
            "Canberra", "Yishun", "Khatib",
            "Yio Chu Kang", "Ang Mo Kio", "Bishan",
        ],
    },
    "North East Line": {
        "stations": [
            "Little India", "Farrer Park", "Boon Keng",
            "Potong Pasir", "Woodleigh", "Serangoon", "Kovan",
        ],
    },
    "Circle Line": {
        "stations": [
            "Bishan", "Lorong Chuan", "Serangoon",
            "Bartley", "Tai Seng", "MacPherson",
            "Paya Lebar", "Dakota", "Mountbatten",
        ],
    },
    "Downtown Line": {
        "stations": [
            "Little India", "Jalan Besar", "Bendemeer",
            "Geylang Bahru", "Mattar", "MacPherson",
            "Ubi", "Kaki Bukit", "Bedok North",
        ],
    },
    "East West Line": {
        "stations": [
            "Aljunied", "Paya Lebar", "Eunos",
            "Kembangan", "Bedok", "Tanah Merah",
            "Simei", "Tampines", "Pasir Ris",
        ],
    },
    "Changi Airport Line": {
        "stations": ["Tanah Merah", "Expo", "Changi Airport"],
    },
}

# ─────────────────────────────────────────────
# MANUAL LABEL OFFSETS – Map 2
# Absolute plot-unit offsets (no scale factor applied).
# Positive x = right, positive y = up.
# ─────────────────────────────────────────────
LABEL_OFFSETS_MAP2 = {
    # ── Interchange stations ─────────────────────────────────────
    "MacPherson":    ( 0.00,  0.85),   # directly above
    "Paya Lebar":    (-0.80, -0.55),   # left-below
    "Serangoon":     ( 0.85,  0.25),   # right
    "Bishan":        (-0.90,  0.10),   # left
    "Little India":  (-0.85, -0.45),   # left-below
    "Tanah Merah":   ( 0.85, -0.50),   # right-below

    # ── Downtown Line central ────────────────────────────────────
    "Mattar":        ( 0.80,  0.45),   # right-above
    "Geylang Bahru": (-0.65,  0.70),   # left-above
    "Bendemeer":     (-0.82,  0.10),   # left
    "Jalan Besar":   (-0.82, -0.30),   # left-below
    "Boon Keng":     (-0.80,  0.22),   # left
    "Farrer Park":   (-0.85,  0.20),   # left
    "Ubi":           ( 0.72,  0.28),   # right
    "Kaki Bukit":    ( 0.72,  0.48),   # right-above
    "Bedok North":   ( 0.78,  0.42),   # right-above

    # ── Circle Line ──────────────────────────────────────────────
    "Tai Seng":      ( 0.72,  0.38),   # right
    "Bartley":       ( 0.72,  0.48),   # right-above
    "Lorong Chuan":  ( 0.85,  0.25),   # right
    "Dakota":        ( 0.72, -0.52),   # right-below
    "Mountbatten":   ( 0.72, -0.52),   # right-below

    # ── North East Line ──────────────────────────────────────────
    "Potong Pasir":  (-0.78,  0.30),   # left
    "Woodleigh":     (-0.78,  0.30),   # left
    "Kovan":         ( 0.72,  0.30),   # right

    # ── North South Line ─────────────────────────────────────────
    "Ang Mo Kio":    (-0.78,  0.25),   # left
    "Yio Chu Kang":  (-0.90,  0.25),   # left
    "Khatib":        (-0.72,  0.25),   # left
    "Yishun":        ( 0.68,  0.25),   # right
    "Canberra":      (-0.68,  0.25),   # left

    # ── East West Line ───────────────────────────────────────────
    "Aljunied":      (-0.72,  0.25),   # left
    "Eunos":         ( 0.10, -0.68),   # below
    "Kembangan":     ( 0.72, -0.50),   # right-below
    "Bedok":         ( 0.68, -0.50),   # right-below
    "Simei":         ( 0.10,  0.68),   # above
    "Tampines":      ( 0.10,  0.68),   # above
    "Pasir Ris":     (-0.32,  0.68),   # above-left

    # ── Changi Airport Line ──────────────────────────────────────
    "Expo":          ( 0.68,  0.30),   # right
    "Changi Airport":( 0.40,  0.68),   # above-right
}


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def load_coordinates(csv_path):
    df = pd.read_csv(csv_path)
    coord = df.set_index("station")[["latitude", "longitude"]].to_dict("index")
    return coord


def haversine_km(coord, s1, s2):
    R = 6371.0088
    lat1 = np.radians(coord[s1]["latitude"])
    lon1 = np.radians(coord[s1]["longitude"])
    lat2 = np.radians(coord[s2]["latitude"])
    lon2 = np.radians(coord[s2]["longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def project_positions(coord, station_list, scale=18.0, min_dist=0.70):
    """
    Equirectangular projection.
    scale    : longer axis of the bounding box in plot units.
    min_dist : minimum spacing between any two nodes (plot units).
    """
    lats = np.array([coord[s]["latitude"]  for s in station_list])
    lons = np.array([coord[s]["longitude"] for s in station_list])

    lat_mid = np.radians(lats.mean())
    x_raw   = np.radians(lons) * np.cos(lat_mid)
    y_raw   = np.radians(lats)

    s = scale / max(x_raw.max() - x_raw.min(),
                    y_raw.max() - y_raw.min())

    x_norm = (x_raw - x_raw.min()) * s
    y_norm = (y_raw - y_raw.min()) * s

    pos = {st: [float(x_norm[i]), float(y_norm[i])]
           for i, st in enumerate(station_list)}

    # Iterative minimum-spacing repulsion
    for _ in range(300):
        moved = False
        keys = list(pos.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                dx = pos[b][0] - pos[a][0]
                dy = pos[b][1] - pos[a][1]
                d  = np.hypot(dx, dy)
                if d < min_dist and d > 1e-9:
                    push = (min_dist - d) / 2
                    nx_ = dx / d * push
                    ny_ = dy / d * push
                    pos[a][0] -= nx_
                    pos[a][1] -= ny_
                    pos[b][0] += nx_
                    pos[b][1] += ny_
                    moved = True
        if not moved:
            break

    return {st: (float(v[0]), float(v[1])) for st, v in pos.items()}


def build_graph(line_definitions, coord):
    G = nx.Graph()
    for line_name, info in line_definitions.items():
        for s1, s2 in zip(info["stations"], info["stations"][1:]):
            km    = round(haversine_km(coord, s1, s2), 2)
            miles = round(km * 0.621371, 2)
            if not G.has_edge(s1, s2):
                G.add_edge(s1, s2,
                           km=km, miles=miles,
                           color=COLORS[line_name],
                           style=STYLES[line_name],
                           line=line_name)
    return G


def auto_label_positions(G, positions, offset=0.45):
    """
    Place each label opposite to the mean neighbour direction.
    Used for Map 1.
    """
    label_pos = {}
    for node in G.nodes():
        nx_pos, ny_pos = positions[node]
        neighbors = list(G.neighbors(node))
        if not neighbors:
            label_pos[node] = (nx_pos, ny_pos - offset)
            continue
        vx, vy = 0.0, 0.0
        for nb in neighbors:
            dx = positions[nb][0] - nx_pos
            dy = positions[nb][1] - ny_pos
            length = np.hypot(dx, dy) or 1.0
            vx += dx / length
            vy += dy / length
        mag = np.hypot(vx, vy) or 1.0
        label_pos[node] = (nx_pos - (vx / mag) * offset,
                           ny_pos - (vy / mag) * offset)
    return label_pos


def smart_label_positions(G, positions, label_offsets=None, offset=0.72):
    """
    Map 2 label placement.
    Uses manual offsets (absolute plot units) for listed stations;
    auto-direction fallback for the rest.
    """
    label_pos = {}
    for node in G.nodes():
        nx_pos, ny_pos = positions[node]
        if label_offsets and node in label_offsets:
            dx, dy = label_offsets[node]
            label_pos[node] = (nx_pos + dx, ny_pos + dy)
        else:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                label_pos[node] = (nx_pos, ny_pos - offset)
                continue
            vx, vy = 0.0, 0.0
            for nb in neighbors:
                dx2 = positions[nb][0] - nx_pos
                dy2 = positions[nb][1] - ny_pos
                length = np.hypot(dx2, dy2) or 1.0
                vx += dx2 / length
                vy += dy2 / length
            mag = np.hypot(vx, vy) or 1.0
            label_pos[node] = (nx_pos - (vx / mag) * offset,
                               ny_pos - (vy / mag) * offset)
    return label_pos


# =====================================================================
# TASK 2 – NETWORK STATISTICS
# =====================================================================

def task2_statistics(G):
    km_values    = [d["km"]    for _, _, d in G.edges(data=True)]
    miles_values = [d["miles"] for _, _, d in G.edges(data=True)]

    total_km    = round(float(np.sum(km_values)),    2)
    total_miles = round(float(np.sum(miles_values)), 2)
    avg_km      = round(float(np.mean(km_values)),   2)
    avg_miles   = round(float(np.mean(miles_values)),2)

    stats = pd.DataFrame({
        "Metric":  ["Total network length", "Average distance per edge"],
        "km":      [total_km,   avg_km],
        "miles":   [total_miles, avg_miles],
    })
    return stats


# =====================================================================
# MAP 1 – DRAWING
# =====================================================================

def draw_edge_labels_map1(ax, G, positions, distance_attr, unit_text):
    """Edge distance labels for Map 1."""
    for s1, s2, data in G.edges(data=True):
        x1, y1 = positions[s1]
        x2, y2 = positions[s2]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy) or 1.0
        px, py = -dy / length, dx / length
        ax.text(
            mx + px * 0.22,
            my + py * 0.22,
            f"{data[distance_attr]:.1f} {unit_text}",
            fontsize=8.5,
            ha="center", va="center",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.12",
                      fc=BACKGROUND, ec="none", alpha=0.92),
            zorder=5,
        )


def draw_map(G, positions, line_definitions, distance_attr,
             unit_text, title, output_path):
    """Map 1: straight edges, compact layout."""
    label_positions = auto_label_positions(G, positions, offset=0.45)

    # Auto-size figure to data extent
    all_x = [positions[n][0] for n in G.nodes()]
    all_y = [positions[n][1] for n in G.nodes()]
    x_span = max(all_x) - min(all_x)
    y_span = max(all_y) - min(all_y)
    base_h = 8.0
    fig_w  = base_h * (x_span / y_span) + 2.0
    fig_h  = base_h + 1.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    # ── Edges ───────────────────────────────────────────────────
    for line_name, info in line_definitions.items():
        edgelist = [
            (s1, s2)
            for s1, s2 in zip(info["stations"], info["stations"][1:])
            if G.has_edge(s1, s2)
        ]
        nx.draw_networkx_edges(
            G, positions, edgelist=edgelist,
            edge_color=COLORS[line_name],
            style=STYLES[line_name],
            width=4.5, ax=ax,
        )

    draw_edge_labels_map1(ax, G, positions, distance_attr, unit_text)

    # ── Nodes ────────────────────────────────────────────────────
    node_line_sets = {n: set() for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        node_line_sets[u].add(d["line"])
        node_line_sets[v].add(d["line"])

    for node in G.nodes():
        interchange = len(node_line_sets[node]) > 1
        if interchange:
            nc, ns, ec, lw = "#DDDDDD", 480, "#666666", 2.2
        else:
            nc = list(G.edges(node, data=True))[0][2]["color"]
            ns, ec, lw = 280, "white", 1.8
        nx.draw_networkx_nodes(
            G, positions, nodelist=[node],
            node_color=nc, node_size=ns,
            edgecolors=ec, linewidths=lw, ax=ax,
        )

    # ── Labels ───────────────────────────────────────────────────
    nx.draw_networkx_labels(
        G, label_positions,
        font_size=9.5, font_weight="bold",
        font_color="#1A1A1A", ax=ax,
    )

    # ── Legend ───────────────────────────────────────────────────
    seen = set()
    for line_name in line_definitions:
        if line_name not in seen:
            ax.plot([], [],
                    color=COLORS[line_name],
                    linestyle=STYLES[line_name],
                    linewidth=3, label=line_name)
            seen.add(line_name)

    ax.legend(
        loc="lower right", frameon=True,
        facecolor="white", edgecolor="#888888",
        fontsize=9.5, title="Key", title_fontsize=11,
    )

    # ── Tight bounds ─────────────────────────────────────────────
    margin = 0.9
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=1.2)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {output_path}")


# =====================================================================
# MAP 2 – DRAWING (spline curves + smart labels)
# =====================================================================

def catmull_rom_chain(points, n_seg=80):
    """Smooth Catmull-Rom spline. Pure numpy."""
    if len(points) < 2:
        return np.array([p[0] for p in points]), np.array([p[1] for p in points])
    if len(points) == 2:
        return (np.linspace(points[0][0], points[1][0], n_seg),
                np.linspace(points[0][1], points[1][1], n_seg))

    pts = np.array([points[0]] + list(points) + [points[-1]], dtype=float)
    all_x, all_y = [], []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        include_end = (i == len(pts) - 3)
        t  = np.linspace(0, 1, n_seg, endpoint=include_end)
        t2 = t * t
        t3 = t2 * t
        x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t +
                    (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 +
                    (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
        y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t +
                    (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 +
                    (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
    return np.array(all_x), np.array(all_y)


def draw_distance_labels_map2(ax, G, positions, distance_attr, unit_text):
    """Edge distance labels for Map 2 — perpendicular offset."""
    for s1, s2, data in G.edges(data=True):
        x1, y1 = positions[s1]
        x2, y2 = positions[s2]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy) or 1.0
        px, py = -dy / length, dx / length
        ax.text(
            mx + px * 0.28,
            my + py * 0.28,
            f"{data[distance_attr]:.1f} {unit_text}",
            fontsize=7,
            ha="center", va="center",
            color="#444444",
            bbox=dict(boxstyle="round,pad=0.09",
                      fc=BACKGROUND, ec="none", alpha=0.90),
            zorder=5,
        )


def draw_map2(G, positions, line_definitions, distance_attr,
              unit_text, title, output_path, label_offsets=None):
    """Map 2: Catmull-Rom curves, MRT-style nodes, smart labels."""

    fig, ax = plt.subplots(figsize=(26, 20), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    # ── 1. Spline curves ────────────────────────────────────────
    draw_order = sorted(
        line_definitions.keys(),
        key=lambda ln: (1 if STYLES[ln] == "dashed" else 0)
    )
    for line_name in draw_order:
        info  = line_definitions[line_name]
        pts   = [positions[s] for s in info["stations"]]
        xs, ys = catmull_rom_chain(pts, n_seg=80)
        color = COLORS[line_name]

        if line_name == "Changi Airport Line":
            ax.plot(xs, ys, color="white",  linewidth=12.0,
                    solid_capstyle="round", zorder=2)
            ax.plot(xs, ys, color=color,    linewidth=9.0,
                    solid_capstyle="round", zorder=3)
            ax.plot(xs, ys, color="white",  linewidth=4.0,
                    solid_capstyle="round", zorder=4)
        else:
            ax.plot(xs, ys, color="white",  linewidth=9.0,
                    solid_capstyle="round", zorder=2)
            ax.plot(xs, ys, color=color,    linewidth=5.5,
                    solid_capstyle="round", zorder=3)

    # ── 2. Edge distance labels ──────────────────────────────────
    draw_distance_labels_map2(ax, G, positions, distance_attr, unit_text)

    # ── 3. Nodes ─────────────────────────────────────────────────
    node_line_sets = {n: set() for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        node_line_sets[u].add(d["line"])
        node_line_sets[v].add(d["line"])

    for node in G.nodes():
        x, y = positions[node]
        is_interchange = len(node_line_sets[node]) > 1
        if is_interchange:
            ax.plot(x, y, "o", markersize=16, color="white",
                    markeredgecolor="#555555", markeredgewidth=2.5, zorder=6)
        else:
            line_name  = list(node_line_sets[node])[0]
            node_color = COLORS[line_name]
            ax.plot(x, y, "o", markersize=10, color=node_color,
                    markeredgecolor="white", markeredgewidth=2.0, zorder=6)

    # ── 4. Station labels ────────────────────────────────────────
    lp = smart_label_positions(G, positions,
                                label_offsets=label_offsets,
                                offset=0.72)
    for node in G.nodes():
        ax.text(lp[node][0], lp[node][1], node,
                fontsize=8.5, fontweight="bold",
                color="#1A1A1A", ha="center", va="center",
                zorder=7)

    # ── 5. Legend ────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    seen, legend_handles, legend_labels = set(), [], []
    for line_name in line_definitions:
        if line_name in seen:
            continue
        if line_name == "Changi Airport Line":
            h1 = Line2D([0], [0], color=COLORS[line_name], linewidth=9)
            h2 = Line2D([0], [0], color="white",            linewidth=3.5)
            legend_handles.append((h1, h2))
            legend_labels.append("Changi Airport Line")
            seen.add(line_name)
            continue
        h, = ax.plot([], [], color=COLORS[line_name],
                     linestyle="-", linewidth=3)
        legend_handles.append(h)
        legend_labels.append(line_name)
        seen.add(line_name)

    h_ic, = ax.plot([], [], "o", color="white",
                    markeredgecolor="#555555", markeredgewidth=2,
                    markersize=10, linestyle="None")
    legend_handles.append(h_ic)
    legend_labels.append("Interchange Station")

    ax.legend(
        handles=legend_handles, labels=legend_labels,
        loc="lower right", frameon=True,
        facecolor="white", edgecolor="#AAAAAA",
        fontsize=9, title="Key", title_fontsize=11,
        labelspacing=0.6,
        handler_map={tuple: HandlerTuple()}
    )

    # ── 6. Bounds & layout ───────────────────────────────────────
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    margin = 1.8
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(top=0.95, bottom=0.02, left=0.02, right=0.98)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {output_path}")


# =====================================================================
# USER INPUT
# =====================================================================

def get_distance_choice():
    print("\nSingapore MRT Schematic Network")
    print("Choose the distance unit to display on the graph:")
    print("  1 - Kilometres")
    print("  2 - Miles")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "km", "km"
        if choice == "2":
            return "miles", "mi"
        print("Invalid input. Please enter 1 or 2.")


# =====================================================================
# MAIN
# =====================================================================

def main():
    distance_attr, unit_text = get_distance_choice()

    coord = load_coordinates("station_coordinates.csv")

    # ── MAP 1 ────────────────────────────────────────────────────
    map1_stations = list({s for info in MAP1_LINES.values()
                          for s in info["stations"]})
    pos1 = project_positions(coord, map1_stations, scale=8.0, min_dist=0.50)
    G1   = build_graph(MAP1_LINES, coord)
    draw_map(G1, pos1, MAP1_LINES,
             distance_attr, unit_text,
             title=f"Map 1 \u2013 Original Network  [{unit_text}]",
             output_path="map1.png")

    # ── MAP 2 ────────────────────────────────────────────────────
    map2_stations = list({s for info in MAP2_LINES.values()
                          for s in info["stations"]})
    pos2 = project_positions(coord, map2_stations, scale=20.0, min_dist=1.0)
    G2   = build_graph(MAP2_LINES, coord)

    if not nx.is_connected(G2):
        raise ValueError("Map 2 graph is not connected.")

    draw_map2(G2, pos2, MAP2_LINES,
              distance_attr, unit_text,
              title=f"Map 2 \u2013 Expanded Network  [{unit_text}]",
              output_path="map2.png",
              label_offsets=LABEL_OFFSETS_MAP2)

    # ── TASK 2 ────────────────────────────────────────────────────
    stats = task2_statistics(G2)
    stats.to_csv("task2_summary.csv", index=False)

    print("\n\u2500\u2500 TASK 2 RESULTS (Map 2) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    print(stats.to_string(index=False))
    print("\nFiles created: map1.png  map2.png  task2_summary.csv")


if __name__ == "__main__":
    main()