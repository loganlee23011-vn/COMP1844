import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =====================================================================
# SINGAPORE MRT NETWORK ANALYSIS AND VISUALIZATION SYSTEM
# =====================================================================
# Purpose:
#   - Visualize the Singapore MRT (Mass Rapid Transit) network as a schematic map
#   - Calculate distances between stations using geographic coordinates
#   - Analyze network statistics (total and per-line distances)
#   - Find shortest routes between any two stations using Dijkstra's algorithm
#   - Highlight selected routes on interactive maps
#
# Allowed Libraries:
#   - NumPy: Mathematical calculations (Haversine formula for distances)
#   - Pandas: Data manipulation and CSV file input/output
#   - NetworkX: Graph construction and shortest path computation
#   - Matplotlib: Network visualization and map generation
#
# Core Features:
#   1. Interactive distance unit selection (Kilometres or Miles)
#   2. Network visualization with color-coded lines and stations
#   3. Edge distance calculations (based on geographic coordinates)
#   4. Network-wide statistics (total and average distances)
#   5. Per-line distance statistics
#   6. Shortest path calculation between any two stations (with highlighting)
#   7. Professional schematic maps with legends and labels
#
# Data Sources (station_coordinates.csv):
#   Station geographic coordinates (WGS84 latitude/longitude) were sourced from:
#   [1] Land Transport Authority (LTA) Singapore - Train Station Locations (Geospatial)
#       LTA DataMall Static Datasets:
#       https://datamall.lta.gov.sg/content/datamall/en/static-data.html
#       Dataset ZIP: https://datamall.lta.gov.sg/content/dam/datamall/datasets/Geospatial/TrainStation_Apr2025.zip
#   [2] data.gov.sg - LTA MRT Station Exit (GeoJSON)
#       Land Transport Authority (2019). LTA MRT Station Exit (GEOJSON) (2026) [Dataset].
#       https://data.gov.sg/datasets/d_b39d3a0871985372d7e1637193335da5/view
#   [3] Coordinates cross-verified via Google Maps (https://maps.google.com)
#       and individual Wikipedia station articles.
# =====================================================================

LINE_DEFINITIONS = {
    "East West Line": {
        "stations": [
            "Aljunied", "Paya Lebar", "Eunos", "Kembangan",
            "Bedok", "Tanah Merah", "Simei", "Tampines", "Pasir Ris"
        ],
        "future": 0,
        "color": "#6FCF97",
        "style": "solid"
    },
    "East West Line Branch": {
        "stations": ["Tanah Merah", "Expo", "Changi Airport"],
        "future": 0,
        "color": "#6FCF97",
        "style": "solid"
    },
    "Circle Line": {
        "stations": [
            "Serangoon", "Bartley", "Tai Seng", "MacPherson",
            "Paya Lebar", "Dakota", "Mountbatten", "Stadium",
            "Nicoll Highway", "Promenade"
        ],
        "future": 0,
        "color": "#F6C667",
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
        "color": "#6FA8DC",
        "style": "solid"
    },
    "Downtown Line Extension": {
        "stations": ["Expo", "Xilin", "Sungei Bedok"],
        "future": 1,
        "color": "#6FA8DC",
        "style": "dashed"
    },
    "North East Line": {
        "stations": [
            "Little India", "Farrer Park", "Boon Keng",
            "Potong Pasir", "Woodleigh", "Serangoon", "Kovan"
        ],
        "future": 0,
        "color": "#C39BD3",
        "style": "solid"
    },
    "Thomson-East Coast Line": {
        "stations": [
            "Marine Parade", "Marine Terrace", "Siglap",
            "Bayshore", "Bedok South", "Sungei Bedok"
        ],
        "future": 1,
        "color": "#BCA18A",
        "style": "dashed"
    }
}

# Hex color code (#FF9800 bright orange) used to highlight shortest path edges.
# Provides high contrast and visibility for route emphasis on the network map.
HIGHLIGHT_COLOR = "#FF9800"


# =====================================================================
# CORE UTILITY FUNCTIONS
# =====================================================================


def get_distance_choice():
    """
    Prompts user to select preferred distance unit for display and calculations.
    Returns both the graph attribute key and the display label.
    
    Returns:
        tuple: (distance_attr, unit_text) where:
               - distance_attr (str): 'km' or 'miles' (used as graph edge attribute key)
               - unit_text (str): 'km' or 'mi' (used for display on maps and output)
    """
    print("Singapore MRT Schematic Network")
    print("Choose the distance unit to display and use for route calculation:")
    print("1 - Kilometres")
    print("2 - Miles")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "km", "km"
        if choice == "2":
            return "miles", "mi"
        print("Invalid input. Please enter 1 for kilometres or 2 for miles.")


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculates great-circle distance between two geographic points using the Haversine formula.
    This formula is accurate for Earth as a sphere and accounts for curvature.
    
    Args:
        lat1 (float): Latitude of first point in degrees (range: -90 to 90)
        lon1 (float): Longitude of first point in degrees (range: -180 to 180)
        lat2 (float): Latitude of second point in degrees
        lon2 (float): Longitude of second point in degrees
        
    Returns:
        float: Distance in kilometres between the two coordinates
    """
    earth_radius_km = 6371.0088  # Earth's mean radius in kilometres
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    # Calculate differences in radians
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Apply Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(earth_radius_km * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def build_edge_dataframe(station_df):
    """
    Creates a comprehensive edge dataframe from station coordinates and line definitions.
    Calculates distances between consecutive stations on each MRT line using geographic coordinates.
    
    Args:
        station_df (pd.DataFrame): Input dataframe loaded from station_coordinates.csv.
                                   Coordinates sourced from LTA DataMall (WGS84).
                                   Columns:
                                   - 'station' (str): Station name
                                   - 'latitude' (float): Geographic latitude (WGS84)
                                   - 'longitude' (float): Geographic longitude (WGS84)
        
    Returns:
        pd.DataFrame: Dataframe with columns:
                     - station_1, station_2: Consecutive stations on a line
                     - line: MRT line name
                     - future: Binary flag (0/1)
                     - color: Hex color code
                     - style: Line style (solid/dashed)
                     - km: Distance in kilometres
                     - miles: Distance in miles
    """
    station_lookup = station_df.set_index("station")[["latitude", "longitude"]].to_dict("index")

    rows = []
    # Process each MRT line and calculate distances between consecutive stations
    for line_name, info in LINE_DEFINITIONS.items():
        stations = info["stations"]
        # Iterate through consecutive station pairs on the line
        for station_1, station_2 in zip(stations, stations[1:]):
            lat1 = station_lookup[station_1]["latitude"]
            lon1 = station_lookup[station_1]["longitude"]
            lat2 = station_lookup[station_2]["latitude"]
            lon2 = station_lookup[station_2]["longitude"]
            # Calculate distance using Haversine formula
            km = round(haversine_km(lat1, lon1, lat2, lon2), 3)
            miles = round(km * 0.621371, 3)  # Convert to miles
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
    """
    Constructs an undirected NetworkX graph from the edge dataframe.
    Each edge stores line metadata and both distance measurements for path calculations.
    
    Args:
        edge_df (pd.DataFrame): Edge dataframe from build_edge_dataframe()
        
    Returns:
        nx.Graph: Undirected graph where:
                 - Nodes: Station names
                 - Edges: Connect consecutive stations with metadata attributes:
                   - 'line': MRT line name
                   - 'future': Operational status (0=current, 1=future)
                   - 'color': Display color
                   - 'style': Line style (solid/dashed)
                   - 'km': Distance in kilometres (used as weight)
                   - 'miles': Distance in miles (used as weight)
    """
    graph = nx.Graph()
    # Add edges with all metadata from the dataframe
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
    """
    Defines the X-Y coordinates (schematic positions) for each station on the map.
    These are NOT actual geographic coordinates but rather an optimized schematic layout
    where lines are drawn clearly without overlapping. Coordinates can be manually tuned
    for better visual clarity and label placement.
    
    Returns:
        dict: Mapping of station names (str) to (x, y) coordinate tuples (float, float)
    """
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
    """
    Defines the X-Y offset coordinates for station name labels.
    Labels are positioned slightly offset from node positions to avoid overlapping
    with network edges and maintain readability on the schematic map.
    
    Returns:
        dict: Mapping of station names (str) to label (x, y) coordinate tuples (float, float)
    """
    return {
        "Mattar": (0.0, -0.68),
        "MacPherson": (2.2, 2.78),
        "Ubi": (4.4, 4.98),
        "Kaki Bukit": (6.6, 7.20),
        "Aljunied": (-4.15, -2.22),
        "Paya Lebar": (2.2, -2.90),
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
        "Expo": (18.0, -4.10),
        "Changi Airport": (22.95, -3.25),
        "Xilin": (18.55, -5.32),
        "Sungei Bedok": (14.8, -10.05),
        "Bedok South": (19.55, -6.60),
        "Bartley": (-2.7, 6.08),
        "Serangoon": (-5.2, 7.90),
        "Mountbatten": (3.5, -6.70),
        "Stadium": (4.9, -8.40),
        "Nicoll Highway": (5.8, -10.10),
        "Promenade": (8.9, -11.80),
        "Farrer Park": (-10.95, 0.28),
        "Boon Keng": (-8.55, 2.28),
        "Potong Pasir": (-6.35, 4.30),
        "Woodleigh": (-5.15, 6.36),
        "Kovan": (-2.55, 9.68),
        "Marine Parade": (11.2, -12.90),
        "Marine Terrace": (14.4, -11.48),
        "Siglap": (16.1, -9.86),
        "Bayshore": (17.75, -8.00)
    }


def draw_network(graph, edge_df, distance_attr, output_path, highlighted_path=None):
    """
    Renders the MRT network as a professional schematic map with all visual elements.
    Supports optional highlighting of a specific route through the network.
    
    Args:
        graph (nx.Graph): NetworkX graph from build_graph() with station nodes and edges
        edge_df (pd.DataFrame): Edge dataframe containing line and distance information
        distance_attr (str): Distance attribute key to display on edges ('km' or 'miles')
        output_path (str): File path where the PNG map will be saved
        highlighted_path (list, optional): List of station names forming the shortest path.
                                          If provided, edges in this path are drawn in
                                          orange with wider lines for emphasis.
        
    Features:
        - Distinct colors for each MRT line
        - Interchange stations (multiple lines) vs single-line stations styled differently
        - Distance labels on each edge
        - Professional legend showing all lines
        - Highlighted edges for shortest route (if provided)
        - Yellow node highlighting for stations in the shortest path
        - High-resolution output (300 DPI)
    """
    pos = build_positions()
    label_pos = build_label_positions()

    # Create figure with main network area and side legend panel
    fig = plt.figure(figsize=(22, 14), facecolor="#eeeeee")
    ax = fig.add_axes([0.04, 0.07, 0.76, 0.86])
    ax.set_facecolor("#eeeeee")

    # Create legend/key panel on the right side of the map
    key_ax = fig.add_axes([0.81, 0.09, 0.17, 0.22])
    key_ax.set_facecolor("white")
    for spine in key_ax.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(1.0)
    key_ax.set_xticks([])
    key_ax.set_yticks([])
    key_ax.set_xlim(0, 1)
    key_ax.set_ylim(0, 1)

    fig.text(
        0.04, 0.965, "Singapore MRT Schematic Network",
        fontsize=15, fontweight="bold", color="#222222", ha="left", va="top"
    )
    fig.text(
        0.04, 0.944, "Task 1 network map with selectable distance labels",
        fontsize=9.5, color="#555555", ha="left", va="top"
    )
    fig.text(
        0.04, 0.928, f"Displayed edge attribute: {distance_attr}",
        fontsize=9.5, color="#555555", ha="left", va="top"
    )

    # Extract unique MRT lines with their visual properties
    unique_lines = edge_df[["line", "color", "style"]].drop_duplicates()

    # Build set of edges that are part of the shortest path for highlighting (Feature #2)
    highlighted_edge_set = set()
    if highlighted_path is not None and len(highlighted_path) >= 2:
        # Convert path sequence to sorted edge tuples for comparison
        highlighted_edge_set = {
            tuple(sorted((u, v))) for u, v in zip(highlighted_path[:-1], highlighted_path[1:])
        }

    # Draw all edges line by line with their respective colors and styles
    for _, line_row in unique_lines.iterrows():
        line_name = line_row["line"]
        # Get all edges belonging to this MRT line
        all_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d["line"] == line_name
        ]

        # Separate edges into normal (non-highlighted) and bright (highlighted) categories
        normal_edges = []
        bright_edges = []

        for u, v in all_edges:
            edge_key = tuple(sorted((u, v)))
            if edge_key in highlighted_edge_set:
                bright_edges.append((u, v))
            else:
                normal_edges.append((u, v))

        # Adjust line width based on operational status (solid=thicker, dashed=thinner)
        base_width = 4.8 if line_row["style"] == "solid" else 3.3

        # Draw normal edges with reduced opacity
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=normal_edges,
            edge_color=line_row["color"],
            width=base_width - 0.5,
            style=line_row["style"],
            alpha=0.6,
            ax=ax
        )

        # Draw highlighted edges (shortest path) with increased width and full opacity for emphasis
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=bright_edges,
            edge_color=line_row["color"],
            width=base_width + 3.5,
            style="solid",
            alpha=1.0,
            ax=ax
        )

    # Create and display distance labels on all edges
    edge_labels = {}
    suffix = "km" if distance_attr == "km" else "mi"
    for u, v, d in graph.edges(data=True):
        edge_labels[(u, v)] = f"{round(d[distance_attr], 1)}{suffix}"

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=6.6,
        font_color="#444444",
        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75),
        rotate=False,
        ax=ax
    )

    # Identify interchange stations (multiple lines) vs single-line stations for distinct styling
    node_lines = {node: set() for node in graph.nodes()}
    for u, v, d in graph.edges(data=True):
        node_lines[u].add(d["line"])
        node_lines[v].add(d["line"])

    # Separate nodes by number of connected lines
    interchange_nodes = [node for node, lines in node_lines.items() if len(lines) > 1]
    single_nodes = [node for node, lines in node_lines.items() if len(lines) == 1]

    # Draw interchange stations in neutral gray color (larger size for visibility)
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=interchange_nodes,
        node_color="#f3f3f3",
        node_size=400,
        edgecolors="#9a9a9a",
        linewidths=1.8,
        ax=ax
    )

    # Draw single-line stations in their respective line color for visual consistency
    for node in single_nodes:
        line_name = list(node_lines[node])[0]
        line_color = unique_lines.loc[unique_lines["line"] == line_name, "color"].iloc[0]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=[node],
            node_color=line_color,
            node_size=240,
            edgecolors="white",
            linewidths=1.6,
            ax=ax
        )

    # Highlight stations along shortest path route with distinct yellow color (Feature #2)
    if highlighted_path is not None and len(highlighted_path) >= 2:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=highlighted_path,
            node_color="#FFE599",  # Bright yellow for visibility
            node_size=320,
            edgecolors=HIGHLIGHT_COLOR,  # Orange border for emphasis
            linewidths=2.2,
            ax=ax
        )

    nx.draw_networkx_labels(
        graph,
        label_pos,
        font_size=7.1,
        font_color="#111111",
        font_weight="normal",
        ax=ax
    )

    # Add legend title
    key_ax.text(0.08, 0.94, "Key", fontsize=10.0, fontweight="bold", color="#222222", ha="left", va="top")

    # Build legend entries showing all available MRT lines with colors and styles
    legend_items = [
    (line_name, info["color"], info["style"])
    for line_name, info in LINE_DEFINITIONS.items()
    ]

    # Draw legend items (colored lines with labels)
    for i, (name, color, style) in enumerate(legend_items):
        y = 0.84 - i * 0.10
        key_ax.plot([0.08, 0.36], [y, y], color=color, linewidth=2.8, linestyle=style)
        key_ax.text(0.42, y, name, fontsize=7.2, color="#222222", va="center", ha="left")

    # Configure axis appearance and save figure at high resolution
    ax.set_xlim(-13.8, 23.8)
    ax.set_ylim(-15.4, 10.4)
    ax.set_aspect("equal")  # Maintain square aspect ratio for accurate schematic
    ax.axis("off")  # Hide axis spines and ticks

    # Save as high-resolution PNG image
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # Display the map interactively in a window
    plt.show()
    plt.close(fig)


def task2_statistics(edge_df):
    """
    Calculates network-wide statistics: total and average distances across all edges.
    This is Task 2 of the coursework requirements.
    
    Args:
        edge_df (pd.DataFrame): Edge dataframe with 'km' and 'miles' columns
        
    Returns:
        pd.DataFrame: Summary statistics with metrics for both distance units
                     - Total network length (sum of all edges)
                     - Average distance per edge (mean of all edges)
    """
    total_km = round(edge_df["km"].sum(), 3)
    total_miles = round(edge_df["miles"].sum(), 3)
    avg_km = round(edge_df["km"].mean(), 3)
    avg_miles = round(edge_df["miles"].mean(), 3)
    return pd.DataFrame({
        "metric": ["Total network length", "Average distance per edge"],
        "km": [total_km, avg_km],
        "miles": [total_miles, avg_miles]
    })


def line_distance_statistics(edge_df):
    """
    Calculates total distances for each MRT line independently.
    Useful for comparing network coverage and line lengths across the system.
    Additional Feature #3: Total distance per line
    
    Args:
        edge_df (pd.DataFrame): Edge dataframe with 'line', 'km', and 'miles' columns
        
    Returns:
        pd.DataFrame: Summary by line sorted by total kilometres (descending)
                     Shows which lines are longest and most extensive
    """
    line_summary = (
        edge_df.groupby("line", as_index=False)[["km", "miles"]]
        .sum()
        .sort_values("km", ascending=False)
        .reset_index(drop=True)
    )
    line_summary["km"] = line_summary["km"].round(3)
    line_summary["miles"] = line_summary["miles"].round(3)
    return line_summary


def normalise_station_name(user_text, valid_stations):
    """
    Normalizes user input to match valid station names.
    Handles case-insensitivity and extra whitespace automatically.
    
    Args:
        user_text (str): Raw user input for station name
        valid_stations (list): List of correctly formatted station names
        
    Returns:
        str or None: Matched station name if found (with proper formatting),
                    None if no match in valid_stations
    """
    # Normalize input: trim whitespace and convert to lowercase
    cleaned = " ".join(user_text.strip().split()).lower()
    # Create case-insensitive lookup dictionary
    station_lookup = {station.lower(): station for station in valid_stations}
    return station_lookup.get(cleaned)


def choose_station(prompt_text, valid_stations):
    """
    Interactive function to prompt and validate station selection.
    Continues prompting until user enters a valid station name.
    
    Args:
        prompt_text (str): Question/prompt to display to user
        valid_stations (list): List of valid station names for validation
        
    Returns:
        str: Validated station name (with correct formatting from valid_stations)
    """
    while True:
        user_value = input(prompt_text).strip()
        # Normalize input and check against valid stations
        station_name = normalise_station_name(user_value, valid_stations)
        if station_name is not None:
            return station_name
        print("Station not found. Please choose a station from the displayed list.")


def shortest_path_analysis(graph, distance_attr):
    """
    Calculates the shortest path between two user-selected stations using Dijkstra's algorithm.
    Additional Feature #1: Shortest path computation between any two stations
    
    Args:
        graph (nx.Graph): NetworkX graph with distance-weighted edges
        distance_attr (str): Weight attribute for calculation ('km' or 'miles')
        
    Returns:
        tuple: (start_station, end_station, path, total_distance) where:
               - start_station (str): Starting point station name
               - end_station (str): Destination station name
               - path (list): List of station names from start to end (inclusive)
               - total_distance (float): Total route distance in selected unit
    """
    # Get all stations in the graph
    valid_stations = sorted(graph.nodes())
    print("\nAVAILABLE STATIONS")
    print(", ".join(valid_stations))

    # Prompt user to select start and destination stations
    start_station = choose_station("\nEnter the start station: ", valid_stations)
    end_station = choose_station("Enter the destination station: ", valid_stations)

    # Calculate shortest path using specified distance metric
    path = nx.shortest_path(graph, source=start_station, target=end_station, weight=distance_attr)
    total_distance = nx.shortest_path_length(graph, source=start_station, target=end_station, weight=distance_attr)
    return start_station, end_station, path, round(float(total_distance), 3)


def main():
    """
    Main orchestration function for the complete MRT network analysis system.
    Coordinates all operations from data loading through visualization.
    
    Workflow:
        1. User selects distance unit (km or miles)
        2. Load station coordinates from CSV file
        3. Build edge database and NetworkX graph
        4. Generate Task 1: Network schematic map (task1_map.png)
        5. Calculate Task 2: Network statistics (task2_summary.csv)
        6. Additional Feature #3: Per-line statistics (line_distance_summary.csv)
        7. Optional: Task 1 Additional Features #1 & #2:
           - Shortest path calculation between stations
           - Highlighted map showing the route (shortest_path_map.png)
        
    Output Files:
        - task1_map.png: Network schematic with all stations and edges
        - edge_distances.csv: Complete edge list with distances
        - task2_summary.csv: Network-wide statistics (total and average)
        - line_distance_summary.csv: Per-line distance breakdown
        - shortest_path_map.png: Map with highlighted route (if selected)
    """
    # Step 1: Get user preference for distance units
    distance_attr, unit_text = get_distance_choice()

    # Step 2: Load station data and build graph
    coordinates_path = "station_coordinates.csv"
    station_df = pd.read_csv(coordinates_path)
    edge_df = build_edge_dataframe(station_df)
    graph = build_graph(edge_df)

    # Validation: Ensure graph connectivity
    if not nx.is_connected(graph):
        raise ValueError("The generated graph is not connected.")

    # Step 3: Generate Task 1 - Network visualization
    draw_network(graph, edge_df, distance_attr, "task1_map.png")
    edge_df.to_csv("edge_distances.csv", index=False)

    # Step 4: Generate Task 2 - Network statistics
    summary_df = task2_statistics(edge_df)
    summary_df.to_csv("task2_summary.csv", index=False)

    # Step 5: Generate Additional Feature #3 - Per-line statistics
    line_summary_df = line_distance_statistics(edge_df)
    line_summary_df.to_csv("line_distance_summary.csv", index=False)

    # Display results to user
    print("\nTASK 2 RESULTS")
    print(summary_df.to_string(index=False))

    print("\nADDITIONAL FEATURE - TOTAL DISTANCE PER LINE")
    print(line_summary_df.to_string(index=False))

    # Step 6: Optional - Calculate shortest path
    print("\nDo you want to calculate a shortest path and highlight it on the map?")
    print("1 - Yes")
    print("2 - No")

    while True:
        route_choice = input("Enter 1 or 2: ").strip()
        if route_choice in {"1", "2"}:
            break
        print("Invalid input. Please enter 1 or 2.")

    created_files = [
        "task1_map.png",
        "edge_distances.csv",
        "task2_summary.csv",
        "line_distance_summary.csv"
    ]

    # Step 7: If user chose yes, compute and visualize shortest path
    if route_choice == "1":
        # Additional Features #1 & #2: Shortest path with highlighting
        start_station, end_station, path, total_distance = shortest_path_analysis(graph, distance_attr)
        draw_network(
            graph,
            edge_df,
            distance_attr,
            "shortest_path_map.png",
            highlighted_path=path
        )
        print("\nADDITIONAL FEATURE - SHORTEST PATH")
        print(f"Start station: {start_station}")
        print(f"Destination station: {end_station}")
        print("Shortest route:")
        print(" -> ".join(path))
        print(f"Total route distance: {round(total_distance, 2)} {unit_text}")
        created_files.append("shortest_path_map.png")

    # Summary of generated files
    print("\nFiles created:")
    for file_name in created_files:
        print(f"- {file_name}")


if __name__ == "__main__":
    main()