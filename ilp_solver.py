import pulp
import json
import requests
from math import radians, cos, sin, asin, sqrt

# ==========================================
# 1. INPUT DATA (Your JSON Structure)
# ==========================================
places_json = [
    {
        "name": "Mahakaleshwar Temple",
        "category": "Spiritual",
        "latitude": 24.5905075,
        "longitude": 73.6748199,
        "time_duration_hours": 1.5,
        "smart_score": 31.68,
        "native_price":20,
        "schedule": { "monday": "5 AM-10:30 PM" }, # Simplified for Monday trip
        "id": 4
    },
    {
        "name": "City Palace",
        "category": "History",
        "latitude": 24.5764421,
        "longitude": 73.6835109,
        "time_duration_hours": 2.5,
        "smart_score": 22.52,
        "native_price":80,
        "schedule": { "monday": "9 AM-9 PM" },
        "id": 5
    },{
        "name": "Cubbon",
        "category": "History",
        "latitude": 24.5964421,
        "longitude": 73.6835109,
        "time_duration_hours": 1.5,
        "smart_score": 100.52,
        "native_price":19,
        "schedule": { "monday": "10 AM-11 AM, 4 PM-7 PM" },
        "id": 10
    },
    {
        "name": "Fateh Sagar Lake",
        "category": "Nature",
        "latitude": 24.6022,
        "longitude": 73.6743,
        "time_duration_hours": 1.0,
        "smart_score": 25.0,
        "native_price":1000,
        "schedule": { "monday": "8 AM-11 PM" },
        "id": 6
    },
    {
        "name": "Saheliyon-ki-Bari",
        "category": "Garden",
        "latitude": 24.6033,
        "longitude": 73.6933,
        "time_duration_hours": 1.0,
        "smart_score": 20.0,
        "native_price":20,
        "schedule": { "monday": "9 AM-7 PM" },
        "id": 7
    },
    {
        "name": "Jagdish Temple",
        "category": "Spiritual",
        "latitude": 24.5794,
        "longitude": 73.6843,
        "time_duration_hours": 0.5,
        "smart_score": 18.0,
        "native_price":0,
        "schedule": { "monday": "5 AM-10 PM" },
        "id": 8
    },
     {
        "name": "Bagore Ki Haveli",
        "category": "History",
        "latitude": 24.5799,
        "longitude": 73.6811,
        "time_duration_hours": 1.5,
        "smart_score": 24.0,
        "native_price":25,
        "schedule": { "monday": "9:30 AM-5:30 PM" },
        "id": 9
    },
    {
        "name": "ccvv",
        "category": "History",
        "latitude": 24.5799,
        "longitude": 73.6811,
        "time_duration_hours": 3.5,
        "smart_score": 2.0,
        "native_price":20,
        "schedule": { "monday": "9:30 AM-5:30 PM" },
        "id": 11
    }
]

# Hotel coordinates (Starting point)
HOTEL_DATA = {
    "name": "Hotel (Start/End)",
    "latitude": 24.5854, # Example: Central Udaipur
    "longitude": 73.6825,
    "id": 0 # ID 0 is reserved for Hotel
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def parse_time_to_minutes(time_str):
    """Converts '9:30 AM' or '5 PM' to minutes from midnight."""
    time_str = time_str.strip().upper()
    is_pm = "PM" in time_str
    is_am = "AM" in time_str
    clean_time = time_str.replace("AM", "").replace("PM", "").strip()
    
    parts = clean_time.split(":")
    hours = int(parts[0])
    minutes = int(parts[1]) if len(parts) > 1 else 0
    
    if is_pm and hours != 12:
        hours += 12
    if is_am and hours == 12:
        hours = 0
        
    return hours * 60 + minutes

# def haversine(lat1, lon1, lat2, lon2):
#     """Calculates distance in km, then estimates travel time (assume 30 km/h speed)."""
#     R = 6371  # Earth radius in km
#     dlat = radians(lat2 - lat1)
#     dlon = radians(lon2 - lon1)
#     a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a))
#     km = R * c
    
#     # Assume average city speed 30 km/h -> 2 mins per km
#     # Adding a small buffer of 5 mins for parking/traffic
#     travel_time_mins = (km / 30) * 60 + 5 
#     return travel_time_mins
import requests

def chunk_origin_indices(num_nodes, max_pairs, num_destinations):
    """
    Yield ranges of origin indices such that:
        len(chunk) * num_destinations <= max_pairs
    """
    max_origins = max_pairs // num_destinations
    if max_origins < 1:
        raise ValueError("max_pairs too small for the number of destinations")

    start = 0
    while start < num_nodes:
        end = min(start + max_origins, num_nodes)
        yield range(start, end)
        start = end


def build_distance_matrix_ola_batched(all_nodes, api_key, mode="driving", max_pairs=50):
    """
    Build DIST_MATRIX[(id_i, id_j)] = travel_time_minutes for ALL i,j
    using Ola Distance Matrix API with batching on origins.

    - all_nodes: list of dicts with 'id', 'latitude', 'longitude'
    - api_key: your Ola Maps API key
    - mode: "driving", etc.
    - max_pairs: maximum origins * destinations per request (Ola limit ≈ 50)
    """
    ids = [n["id"] for n in all_nodes]
    coords = [(n["latitude"], n["longitude"]) for n in all_nodes]
    num_nodes = len(all_nodes)

    # Build coordinate strings "lat,lon" for all nodes
    coord_strings = [f"{lat},{lon}" for (lat, lon) in coords]

    # Destinations are always ALL nodes
    destinations_param = "|".join(coord_strings)
    num_destinations = num_nodes

    DIST_MATRIX = {}

    url = "https://api.olamaps.io/routing/v1/distanceMatrix"

    # Loop over origin chunks (each chunk small enough so origins * dest <= max_pairs)
    for origin_range in chunk_origin_indices(num_nodes, max_pairs, num_destinations):
        origin_idxs = list(origin_range)
        origin_coord_strings = [coord_strings[i] for i in origin_idxs]
        origins_param = "|".join(origin_coord_strings)

        params = {
            "origins": origins_param,
            "destinations": destinations_param,
            "mode": mode,
            "api_key": api_key,
        }

        resp = requests.get(url, params=params)

        if resp.status_code != 200:
            print("Ola API error for origins indices:", origin_idxs)
            print("Status:", resp.status_code)
            print("Body:", resp.text)
            raise RuntimeError("Failed to fetch distance matrix from Ola Maps")

        data = resp.json()

        if data.get("status") != "SUCCESS":
            print("Ola returned non-SUCCESS status for origins indices", origin_idxs, ":", data)
            raise RuntimeError("Ola Maps Distance Matrix returned non-SUCCESS")

        rows = data["rows"]   # one row per origin in this chunk

        # rows length == len(origin_idxs), each row has 'elements' for all destinations
        for local_i, row in enumerate(rows):
            i_idx = origin_idxs[local_i]   # global index in all_nodes
            id_i = ids[i_idx]

            elements = row["elements"]    # length == num_destinations
            for j_idx, elem in enumerate(elements):
                id_j = ids[j_idx]

                if id_i == id_j:
                    DIST_MATRIX[(id_i, id_j)] = 0.0
                    continue

                if elem.get("status") != "OK":
                    # If Ola cannot route this pair, give it a huge time to discourage this edge
                    DIST_MATRIX[(id_i, id_j)] = 9999.0
                    continue

                # Ola returns duration in SECONDS
                duration_sec = elem["duration"]
                travel_time_mins = duration_sec / 60.0
                DIST_MATRIX[(id_i, id_j)] = travel_time_mins
    print(DIST_MATRIX)
    return DIST_MATRIX



def format_minutes(mins):
    """Converts minutes (e.g., 630) to readable time (10:30 AM)."""
    h = int(mins // 60)
    m = int(mins % 60)
    period = "AM"
    if h >= 12:
        period = "PM"
    if h > 12:
        h -= 12
    if h == 0:
        h = 12
    return f"{h:02d}:{m:02d} {period}"

# ==========================================
# 3. DATA PREPARATION
# ==========================================
# Combine Hotel and Places
all_nodes = [HOTEL_DATA] + places_json
N_nodes = len(all_nodes)
PLACES_INDICES = [p['id'] for p in places_json]
ALL_INDICES = [0] + PLACES_INDICES

# Parameters
SCORES = {p['id']: p['smart_score'] for p in places_json}
DURATIONS = {p['id']: int(p['time_duration_hours'] * 60) for p in places_json}
SCORES[0] = 0
DURATIONS[0] = 0
PRICES = {p['id']: p['native_price'] for p in places_json}
PRICES[0] = 0  # Hotel has no ticket cost


# Time Windows (Extracting Monday for example)
# Handles multiple intervals if "9 AM-1 PM, 2 PM-5 PM" format exists
TIME_WINDOWS = {} 
for node in all_nodes:
    idx = node['id']
    if idx == 0:
        TIME_WINDOWS[0] = [(0, 1440)] # Hotel open 24/7
        continue
        
    schedule_str = node['schedule']['monday'] # Assuming Monday trip
    intervals = []
    # Splitting by comma if multiple intervals exist
    parts = schedule_str.split(',')
    for part in parts:
        start_str, end_str = part.split('-')
        start_min = parse_time_to_minutes(start_str)
        end_min = parse_time_to_minutes(end_str)
        intervals.append((start_min, end_min))
    TIME_WINDOWS[idx] = intervals

K_INDICES = {i: range(len(TIME_WINDOWS[i])) for i in ALL_INDICES}

# # Distance Matrix (Travel Times)
# DIST_MATRIX = {}
# for i in all_nodes:
#     for j in all_nodes:
#         t_time = haversine(i['latitude'], i['longitude'], j['latitude'], j['longitude'])
#         DIST_MATRIX[(i['id'], j['id'])] = t_time


API_KEY = "dY3GvlLELoYW4eXklPI9weknExD8uZRP0BFR73kQ"   # <-- put your actual key

DIST_MATRIX = build_distance_matrix_ola_batched(all_nodes, API_KEY)

# Global Constraints
BIG_M = 10000
LUNCH_START = 12 * 60  # 12:00 PM
LUNCH_END = 14 * 60    # 2:00 PM
FOOD_DUR = 120         # 2 Hours
MAX_DAY_TIME = 20.5 * 60 # Trip must end by 10 PM
WEIGHT_SCORE = 10      # High priority on score
WEIGHT_DIST = 0.1      # Low penalty for distance
WEIGHT_WAIT = 0.5      # Medium penalty for waiting
DAILY_BUDGET = 200  # for example; replace with whatever you want (₹, $ etc.)

# ==========================================
# 4. ILP MODEL FORMULATION
# ==========================================
prob = pulp.LpProblem("Itinerary_Optimizer", pulp.LpMaximize)

# --- Decision Variables ---

# x[i]: 1 if place i is visited
x = pulp.LpVariable.dicts("x", PLACES_INDICES, cat=pulp.LpBinary)

# y[i][j]: Flow from i to j
y = pulp.LpVariable.dicts("y", [(i, j) for i in ALL_INDICES for j in ALL_INDICES if i != j], cat=pulp.LpBinary)

# z[i]: 1 if lunch is taken AFTER node i
z = pulp.LpVariable.dicts("z", ALL_INDICES, cat=pulp.LpBinary)

# w[i][k]: Window selection
w = pulp.LpVariable.dicts("w", [(i, k) for i in PLACES_INDICES for k in K_INDICES[i]], cat=pulp.LpBinary)

# Continuous Time Variables
A = pulp.LpVariable.dicts("Arr_Time", ALL_INDICES, lowBound=0, upBound=1440)
S = pulp.LpVariable.dicts("Start_Time", ALL_INDICES, lowBound=0, upBound=1440)
E = pulp.LpVariable.dicts("End_Time", ALL_INDICES, lowBound=0, upBound=1440)
Wait = pulp.LpVariable.dicts("Wait_Time", PLACES_INDICES, lowBound=0)
BreakStart = pulp.LpVariable("Break_Start", lowBound=0, upBound=1440)
u = pulp.LpVariable.dicts("Order", PLACES_INDICES, lowBound=0, upBound=len(PLACES_INDICES))

# --- Objective Function ---
# Maximize (Total Score) - (Weighted Distance) - (Weighted Waiting Time)
prob += (
    pulp.lpSum([SCORES[i] * x[i] for i in PLACES_INDICES]) 
    - WEIGHT_DIST * pulp.lpSum([DIST_MATRIX[(i, j)] * y[(i, j)] for i in ALL_INDICES for j in ALL_INDICES if i != j])
    - WEIGHT_WAIT * pulp.lpSum([Wait[i] for i in PLACES_INDICES])
)

# --- Constraints ---

# 1. Start and End at Hotel
prob += pulp.lpSum([y[(0, j)] for j in PLACES_INDICES]) == 1
prob += pulp.lpSum([y[(i, 0)] for i in PLACES_INDICES]) == 1

# 2. Flow Conservation
for j in PLACES_INDICES:
    prob += pulp.lpSum([y[(i, j)] for i in ALL_INDICES if i != j]) == x[j]
    prob += pulp.lpSum([y[(j, k)] for k in ALL_INDICES if k != j]) == x[j]

# Total ticket cost of visited places must be within daily budget
prob += pulp.lpSum([PRICES[i] * x[i] for i in PLACES_INDICES]) <= DAILY_BUDGET


# 3. Time Progression & Departure
# Hotel Start of Day (Assume 8:00 AM start for simplicity, or flexible)
prob += E[0] >= 8 * 60 

for i in PLACES_INDICES:
    # E_i = S_i + Dur_i * x_i
    prob += E[i] == S[i] + DURATIONS[i] * x[i]
    # S_i >= A_i (Visit starts after arrival)
    prob += S[i] >= A[i]
    # Wait_i definition
    prob += Wait[i] == S[i] - A[i]

# 4. Time Windows (Big M)
for i in PLACES_INDICES:
    prob += pulp.lpSum([w[(i, k)] for k in K_INDICES[i]]) == x[i] # Pick 1 window if visited
    for k in K_INDICES[i]:
        start_w, end_w = TIME_WINDOWS[i][k]
        prob += S[i] >= start_w - BIG_M * (1 - w[(i, k)])
        prob += E[i] <= end_w + BIG_M * (1 - w[(i, k)])

# 5. Lunch Break Constraints
prob += pulp.lpSum([z[i] for i in ALL_INDICES]) == 1 # Exactly one lunch break location
prob += BreakStart >= LUNCH_START
prob += BreakStart <= LUNCH_END

# Break starts after departure from selected node
for i in ALL_INDICES:
    prob += BreakStart >= E[i] - BIG_M * (1 - z[i])

# 6. Linking Route & Time (The Core Logic)
for i in ALL_INDICES:
    for j in ALL_INDICES:
        if i == j: continue
        
        travel_t = DIST_MATRIX[(i, j)]
        
        # Case A: Lunch taken after i (z[i]=1)
        # A[j] >= BreakStart + 2hrs + Travel
        prob += A[j] >= BreakStart + FOOD_DUR + travel_t - BIG_M*(1 - y[(i, j)]) - BIG_M*(1 - z[i])
        
        # Case B: No Lunch after i (z[i]=0)
        # A[j] >= E[i] + Travel
        prob += A[j] >= E[i] + travel_t - BIG_M*(1 - y[(i, j)]) - BIG_M*z[i]

for i in PLACES_INDICES:
    prob += z[i] <= x[i]

# 7. Subtour Elimination (MTZ)
for i in PLACES_INDICES:
    for j in PLACES_INDICES:
        if i != j:
            prob += u[i] - u[j] + len(PLACES_INDICES) * y[(i, j)] <= len(PLACES_INDICES) - 1

# 8. Max Day Duration (Return to Hotel)
prob += A[0] <= MAX_DAY_TIME

# ==========================================
# 5. SOLVE & OUTPUT
# ==========================================
print("Solving Itinerary...")
status = prob.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 supresses solver logs

if status != pulp.LpStatusOptimal:
    print(status)
    print("No optimal solution found. Try relaxing constraints.")
else:
    print(f"\n✅ Solution Found! Total Objective: {pulp.value(prob.objective):.2f}\n")
    
    # 1. Reconstruct Path
    current_node = 0
    path_data = []
    
    while True:
        path_data.append(current_node)
        
        # Check if lunch happens after this node
        if pulp.value(z[current_node]) > 0.5:
            path_data.append("LUNCH")
            
        # Find next node
        next_node = None
        for j in ALL_INDICES:
            if current_node != j and pulp.value(y[(current_node, j)]) > 0.5:
                next_node = j
                break
        
        if next_node is None or next_node == 0:
            path_data.append(0) # Return to hotel
            break
        current_node = next_node

    # 2. Print Itinerary
    print(f"{'TIME':<15} | {'ACTIVITY':<30} | {'DETAILS'}")
    print("-" * 70)
    
    prev_node = 0 # To track travel time
    
    for idx, item in enumerate(path_data):
    # Handle lunch marker
        if item == "LUNCH":
            bs = pulp.value(BreakStart)
            print(f"{format_minutes(bs):<15} | 🍱 LUNCH BREAK ({FOOD_DUR//60} hrs)   | Rest & Eat")
            print(f"{format_minutes(bs + FOOD_DUR):<15} | 🍱 LUNCH ENDS                | Resume Trip")
            continue

        node_id = item
        node_name = next(n['name'] for n in all_nodes if n['id'] == node_id)

        if node_id == 0:
            # Hotel: first occurrence = DEPART, last occurrence = RETURN
            if idx == 0:
                # Start of day: use E[0] as departure time (8:00 AM in your constraints)
                start_t = pulp.value(E[0])
                print(f"{format_minutes(start_t):<15} | 🏨 DEPART {node_name:<16} | Start Day")
            else:
                # End of day: arrival back at hotel, using A[0]
                arr_t = pulp.value(A[0])
                print(f"{format_minutes(arr_t):<15} | 🏨 RETURN {node_name:<16} | End Day")
        else:
            # Normal place node
            arr_t = pulp.value(A[node_id])
            start_t = pulp.value(S[node_id])
            end_t = pulp.value(E[node_id])
            wait_t = pulp.value(Wait[node_id])

            # Show waiting only if meaningful
            if wait_t > 1:
                print(f"{format_minutes(arr_t):<15} | ⏳ ARRIVE ({int(wait_t)} min wait)    | Waiting for opening...")

            print(f"{format_minutes(start_t):<15} | 📍 VISIT {node_name:<17} | Stay: {DURATIONS[node_id]} mins")