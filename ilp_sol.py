import pulp,os
import json
import requests
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv

load_dotenv() 
# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def parse_time_to_minutes(time_str):
    """Converts '9:30 AM' or '5 PM' to minutes from midnight."""
    if not time_str: return 0
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

def format_minutes(mins):
    """Converts minutes (e.g., 630) to readable time (10:30 AM)."""
    mins = int(round(mins))
    h = (mins // 60) % 24
    m = mins % 60
    period = "AM"
    if h >= 12:
        period = "PM"
    if h > 12:
        h -= 12
    if h == 0:
        h = 12
    return f"{h:02d}:{m:02d} {period}"

def chunk_origin_indices(num_nodes, max_pairs, num_destinations):
    """Yield ranges of origin indices for batching API calls."""
    max_origins = max_pairs // num_destinations
    if max_origins < 1:
        max_origins = 1 
    
    start = 0
    while start < num_nodes:
        end = min(start + max_origins, num_nodes)
        yield range(start, end)
        start = end

def build_distance_matrix_ola_batched(all_nodes, api_key, mode="driving", max_pairs=45):
    """
    Build DIST_MATRIX[(id_i, id_j)] = travel_time_minutes using Ola Maps API.
    """
    ids = [n["id"] for n in all_nodes]
    coords = [(n["latitude"], n["longitude"]) for n in all_nodes]
    num_nodes = len(all_nodes)
    coord_strings = [f"{lat},{lon}" for (lat, lon) in coords]
    destinations_param = "|".join(coord_strings)
    
    DIST_MATRIX = {}
    url = "https://api.olamaps.io/routing/v1/distanceMatrix"

    # Batch processing to respect API limits
    for origin_range in chunk_origin_indices(num_nodes, max_pairs, num_nodes):
        origin_idxs = list(origin_range)
        origin_coord_strings = [coord_strings[i] for i in origin_idxs]
        origins_param = "|".join(origin_coord_strings)

        params = {
            "origins": origins_param,
            "destinations": destinations_param,
            "mode": mode,
            "api_key": api_key,
        }

        try:
            resp = requests.get(url, params=params)
            data = resp.json()
            # print(data)
            if resp.status_code != 200 or data.get("status") != "SUCCESS":
                # print(f"⚠️ Ola API Warning: Batch failed. Using fallback (15m) for batch {origin_idxs}.")
                # print(f"\n🛑 Ola API Error (Batch {origin_idxs}):")
                # print(f"   Status Code: {resp.status_code}")
                # print(f"   Response: {data}")
                # print(f"   Request URL: {resp.url}")  # Helpful to check params
                print("-" * 30)
                for i_idx in origin_idxs:
                    for j_idx in range(num_nodes):
                        DIST_MATRIX[(ids[i_idx], ids[j_idx])] = 15.0 if i_idx != j_idx else 0
                continue

            rows = data["rows"]
            for local_i, row in enumerate(rows):
                i_idx = origin_idxs[local_i]
                id_i = ids[i_idx]
                elements = row["elements"]
                
                for j_idx, elem in enumerate(elements):
                    id_j = ids[j_idx]
                    if id_i == id_j:
                        DIST_MATRIX[(id_i, id_j)] = 0.0
                        continue
                    
                    if elem.get("status") == "OK":
                        DIST_MATRIX[(id_i, id_j)] = elem["duration"] / 60.0
                    else:
                        DIST_MATRIX[(id_i, id_j)] = 9999.0 # Penalize unreachable
        except Exception as e:
            print(f"Error calling Ola API: {e}")
            for i_idx in origin_idxs:
                    for j_idx in range(num_nodes):
                         DIST_MATRIX[(ids[i_idx], ids[j_idx])] = 15.0 if i_idx != j_idx else 0
    print("success")
    return DIST_MATRIX

# ==========================================
# 2. MAIN OPTIMIZATION FUNCTION
# ==========================================

def get_nearby_restaurants(lat, lon, preference='all', radius=1000):
    """
    Finds nearby restaurants using the Overpass API and returns only those with valid addresses.
    
    Args:
        lat (float/str): Latitude of the center point.
        lon (float/str): Longitude of the center point.
        preference (str): 'veg', 'nonveg', or 'all' (default).
        radius (int): Search radius in meters (default 1000).
        
    Returns:
        list: A list of dictionaries [{"name": "...", "address": "..."}, ...]
    """
    
    # 1. Construct the Overpass QL Query
    base_type = '["amenity"="restaurant"]'
    location_filter = f'(around:{radius},{lat},{lon})'
    
    # Define tag filters based on preference
    queries = []
    if preference.lower() == 'veg':
        tags = ['["cuisine"="vegetarian"]', '["diet:vegetarian"="yes"]']
        for tag in tags:
            queries.append(f'node{base_type}{tag}{location_filter};')
            queries.append(f'way{base_type}{tag}{location_filter};')
            
    elif preference.lower() == 'nonveg':
        tags = ['["diet:meat"="yes"]', '["cuisine"~"meat|non_veg"]']
        for tag in tags:
            queries.append(f'node{base_type}{tag}{location_filter};')
            queries.append(f'way{base_type}{tag}{location_filter};')
            
    else: # 'all' or no preference
        queries.append(f'node{base_type}{location_filter};')
        queries.append(f'way{base_type}{location_filter};')

    # Join queries into the full Overpass request structure
    full_query = f"""
    [out:json][timeout:25];
    (
      {''.join(queries)}
    );
    out center;
    """

    overpass_url = "https://overpass-api.de/api/interpreter"
    clean_results = []

    try:
        # 2. Fetch Data
        response = requests.get(overpass_url, params={'data': full_query})
        response.raise_for_status()
        data = response.json()
        
        # 3. Parse and Filter
        for element in data.get('elements', []):
            tags = element.get('tags', {})
            name = tags.get('name')
            
            if not name:
                continue

            # Build address components
            addr_parts = [
                tags.get('addr:housenumber'),
                tags.get('addr:street'),
                tags.get('addr:city'),
                tags.get('addr:postcode')
            ]
            
            # Create address string, removing empty parts
            address_str = ", ".join([p for p in addr_parts if p])
            
            # Check for fallback 'addr:full' if individual parts are missing
            if not address_str:
                address_str = tags.get('addr:full')

            # STRICT FILTER: Only add if we found a valid address
            if address_str:
                clean_results.append({
                    "name": name,
                    "address": address_str
                })
                if len(clean_results) >= 5:
                    break
    except Exception as e:
        print(f"Error fetching restaurants: {e}")
        return []

    return clean_results[:5]

def generate_itinerary(places_json, hotel_name, hotel_lat, hotel_lon, day, daily_budget, pace,start_time,end_time,meal_type):
    """
    Generates an optimized itinerary.
    
    Args:
        places_json (list): List of place dictionaries.
        hotel_name (str): Name of the hotel.
        hotel_lat (float): Latitude of the hotel.
        hotel_lon (float): Longitude of the hotel.
        day (str): Day of the week (e.g., 'monday').
        api_key (str): Ola Maps API Key.
        daily_budget (float): Max budget for tickets.
        pace (str): "slow", "medium", or "fast".

    Returns:
        tuple: (itinerary_json (dict), unvisited_ids (list))
    """
    
    api_key= os.getenv("OLA_MAP_API")
    # 0. Construct Hotel Data Dictionary internally
    hotel_data = {
        "name": hotel_name,
        "latitude": hotel_lat,
        "longitude": hotel_lon,
        "id": 0 # ID 0 is reserved for Hotel
    }
    # 1. Handle Pace Logic
    pace = pace.lower().strip()
    if pace == "relaxed":
        pace_multiplier = 1.25
    elif pace == "moderate":
        pace_multiplier = 1.0
    else:
        pace_multiplier = 0.75

    print(f"🚀 Generating Itinerary | Pace: {pace} (x{pace_multiplier}) | Budget: {daily_budget}")

    # 2. Prepare Data
    day = day.lower().strip()
    all_nodes = [hotel_data] + places_json
    PLACES_INDICES = [p['id'] for p in places_json]
    ALL_INDICES = [hotel_data['id']] + PLACES_INDICES
    
    # Lookups
    SCORES = {p['id']: p['smart_score'] for p in places_json}
    SCORES[hotel_data['id']] = 0
    
    # Apply Pace Multiplier to Duration
    DURATIONS = {p['id']: int(p['time_duration_hours'] * pace_multiplier * 60) for p in places_json}
    DURATIONS[hotel_data['id']] = 0
    
    PRICES = {p['id']: p.get('native_price', 0) for p in places_json}
    PRICES[hotel_data['id']] = 0
    
    # Time Windows
    TIME_WINDOWS = {}
    TIME_WINDOWS[hotel_data['id']] = [(0, 1440)] # Hotel open 24/7
    
    for node in places_json:
        idx = node['id']
        schedule_str = node.get('schedule', {}).get(day, "9 AM-6 PM") 
        intervals = []
        if schedule_str.lower() == "closed":
            schedule_str="4 PM-4 PM"
        parts = schedule_str.split(',')
        for part in parts:
            try:
                start_str, end_str = part.split('-')
                start_min = parse_time_to_minutes(start_str)
                end_min = parse_time_to_minutes(end_str)
                intervals.append((start_min, end_min))
            except:
                intervals.append((540, 1080)) # Default 9-6 if parse fails
        
        TIME_WINDOWS[idx] = intervals

    OPEN_PLACES_INDICES = [i for i in PLACES_INDICES if TIME_WINDOWS.get(i)]
    PLACES_INDICES = OPEN_PLACES_INDICES 
    K_INDICES = {i: range(len(TIME_WINDOWS[i])) for i in ALL_INDICES if i in TIME_WINDOWS}

    # API Call for Distances
    print("Fetching Distance Matrix...")
    DIST_MATRIX = build_distance_matrix_ola_batched(all_nodes, api_key)
    
    # 3. ILP Model Constants
    BIG_M = 10000
    LUNCH_START = 12 * 60  
    LUNCH_END = 14 * 60    
    FOOD_DUR = 90         # 1 Hour Lunch
    MAX_DAY_TIME = end_time  # 10 PM
    WEIGHT_SCORE = 100     
    WEIGHT_DIST = 0.5      
    WEIGHT_WAIT = 0.5      

    # 4. Define Problem
    prob = pulp.LpProblem("Itinerary_Optimizer", pulp.LpMaximize)

    # Variables
    x = pulp.LpVariable.dicts("x", PLACES_INDICES, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", [(i, j) for i in ALL_INDICES for j in ALL_INDICES if i != j], cat=pulp.LpBinary)
    z = pulp.LpVariable.dicts("z", ALL_INDICES, cat=pulp.LpBinary)
    w = pulp.LpVariable.dicts("w", [(i, k) for i in PLACES_INDICES for k in K_INDICES[i]], cat=pulp.LpBinary)

    A = pulp.LpVariable.dicts("Arr_Time", ALL_INDICES, lowBound=0, upBound=1440)
    S = pulp.LpVariable.dicts("Start_Time", ALL_INDICES, lowBound=0, upBound=1440)
    E = pulp.LpVariable.dicts("End_Time", ALL_INDICES, lowBound=0, upBound=1440)
    Wait = pulp.LpVariable.dicts("Wait_Time", PLACES_INDICES, lowBound=0)
    BreakStart = pulp.LpVariable("Break_Start", lowBound=0, upBound=1440)
    u = pulp.LpVariable.dicts("Order", PLACES_INDICES, lowBound=0, upBound=len(PLACES_INDICES))

    # Objective: Maximize Score - Penalties
    prob += (
        pulp.lpSum([SCORES[i] * x[i] * WEIGHT_SCORE for i in PLACES_INDICES]) 
        - WEIGHT_DIST * pulp.lpSum([DIST_MATRIX.get((i, j), 999) * y[(i, j)] for i in ALL_INDICES for j in ALL_INDICES if i != j])
        - WEIGHT_WAIT * pulp.lpSum([Wait[i] for i in PLACES_INDICES])
    )

    # --- Constraints ---
    
    # 1. Route Connectivity
    prob += pulp.lpSum([y[(hotel_data['id'], j)] for j in PLACES_INDICES]) == 1
    prob += pulp.lpSum([y[(i, hotel_data['id'])] for i in PLACES_INDICES]) == 1

    for j in PLACES_INDICES:
        prob += pulp.lpSum([y[(i, j)] for i in ALL_INDICES if i != j]) == x[j]
        prob += pulp.lpSum([y[(j, k)] for k in ALL_INDICES if k != j]) == x[j]

    # 2. Budget Constraint (New)
    prob += pulp.lpSum([PRICES[i] * x[i] for i in PLACES_INDICES]) <= daily_budget

    # 3. Time Constraints
    prob += E[hotel_data['id']] >= start_time  # Start day at 8 AM
    
    for i in PLACES_INDICES:
        # Duration uses the paced duration
        prob += E[i] == S[i] + DURATIONS[i] * x[i]
        prob += S[i] >= A[i]
        prob += Wait[i] == S[i] - A[i]
        
        # Time Windows
        prob += pulp.lpSum([w[(i, k)] for k in K_INDICES[i]]) == x[i]
        for k in K_INDICES[i]:
            start_w, end_w = TIME_WINDOWS[i][k]
            prob += S[i] >= start_w - BIG_M * (1 - w[(i, k)])
            prob += E[i] <= end_w + BIG_M * (1 - w[(i, k)])

    # 4. Lunch Logic
    prob += pulp.lpSum([z[i] for i in ALL_INDICES]) == 1
    prob += BreakStart >= LUNCH_START
    prob += BreakStart <= LUNCH_END
    for i in ALL_INDICES:
        prob += BreakStart >= E[i] - BIG_M * (1 - z[i])
        if i in PLACES_INDICES:
            prob += z[i] <= x[i]

    # 5. Travel Time & Sequencing
    for i in ALL_INDICES:
        for j in ALL_INDICES:
            if i == j: continue
            travel_t = DIST_MATRIX.get((i, j), 30)
            
            # If Lunch after i
            prob += A[j] >= BreakStart + FOOD_DUR + travel_t - BIG_M*(1 - y[(i, j)]) - BIG_M*(1 - z[i])
            # If No Lunch after i
            prob += A[j] >= E[i] + travel_t - BIG_M*(1 - y[(i, j)]) - BIG_M*z[i]

    # 6. Subtour Elimination
    for i in PLACES_INDICES:
        for j in PLACES_INDICES:
            if i != j:
                prob += u[i] - u[j] + len(PLACES_INDICES) * y[(i, j)] <= len(PLACES_INDICES) - 1
                
    prob += A[hotel_data['id']] <= MAX_DAY_TIME
    print("chek2")
    # 5. Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # 6. Construct Results
    timeline = {}
    visited_ids = set()
    print("updating...")
    if status == pulp.LpStatusOptimal:
        current_node = hotel_data['id']
        
        while True:
            # Check for Lunch
            if pulp.value(z[current_node]) and pulp.value(z[current_node]) > 0.5:
                bs = pulp.value(BreakStart)
                be = bs + FOOD_DUR
                time_key = f"{format_minutes(bs)}-{format_minutes(be)}"
                timeline[time_key] = {
                    "name": "Lunch Break",
                    "category": "Food",
                    "latitude": 0, "longitude": 0,
                    "price": 0,
                    "time_duration_hours": FOOD_DUR/60,
                    "id": "LUNCH_BREAK",
                    "description": "Time for lunch and rest."
                }
            
            # Find next node
            next_node = None
            for j in ALL_INDICES:
                if current_node != j and pulp.value(y[(current_node, j)]) > 0.5:
                    next_node = j
                    break
            
            if next_node is None or next_node == hotel_data['id']:
                break
            
            # Process Next Node
            visited_ids.add(next_node)
            start_t = pulp.value(S[next_node])
            end_t = pulp.value(E[next_node])
            time_key = f"{format_minutes(start_t)}-{format_minutes(end_t)}"
            
            # Retrieve object
            place_obj = next(p for p in places_json if p['id'] == next_node)
            
            # Add to itinerary with the ACTUAL scheduled duration
            timeline[time_key] = {
                "name": place_obj['name'],
                "category": place_obj['category'],
                "start_time": format_minutes(start_t), # <--- Added
                "end_time": format_minutes(end_t),
                "latitude": place_obj['latitude'],
                "longitude": place_obj['longitude'],
                "price": place_obj['price'],
                "smart_score": place_obj['smart_score'],
                "id": place_obj['id'],
                "time_duration_hours":round( DURATIONS[next_node] / 60.0,2), # The paced duration
            }
            
            current_node = next_node
    else:
        print("Optimization failed or infeasible (e.g., budget too low).")

    
    restaurant_data = {
        "breakfast": [],
        "lunch": [],
        "dinner": []
    }

    # Sort keys to know order of events
    sorted_keys = sorted(
        timeline.keys(), 
        key=lambda k: parse_time_to_minutes(k.split('-')[0])
    )

    if sorted_keys:
        # --- A. Breakfast (Near First Place) ---
        first_key = sorted_keys[0]
        first_place = timeline[first_key]
        
        # Only suggest breakfast if the first place is NOT lunch
        if first_place['id'] != "LUNCH_BREAK":
            print(f"  > Breakfast near: {first_place['name']}")
            restaurant_data["breakfast"] = get_nearby_restaurants(
                first_place['latitude'], 
                first_place['longitude'], 
                preference=meal_type, 
                radius=1000
            )

        # --- B. Lunch (Near Place Before Lunch) ---
        lunch_key = next((k for k in sorted_keys if timeline[k]['id'] == "LUNCH_BREAK"), None)
        
        if lunch_key:
            lunch_index = sorted_keys.index(lunch_key)
            
            # Logic: Lunch should be near the place we just finished visiting
            if lunch_index > 0:
                prev_key = sorted_keys[lunch_index - 1]
                prev_place = timeline[prev_key]
                
                print(f"  > Lunch near: {prev_place['name']}")
                
                # 1. Fetch Restaurants
                restaurant_data["lunch"] = get_nearby_restaurants(
                    prev_place['latitude'], 
                    prev_place['longitude'], 
                    preference=meal_type, 
                    radius=1000
                )
                
                # 2. Update the "Lunch Break" timeline item with location data
                # (This helps the frontend show the lunch marker on the map)
                timeline[lunch_key]['latitude'] = prev_place['latitude']
                timeline[lunch_key]['longitude'] = prev_place['longitude']
                
            else:
                # If Lunch is the very first item, search near Hotel
                print("  > Lunch near Hotel")
                restaurant_data["lunch"] = get_nearby_restaurants(hotel_lat, hotel_lon,meal_type, radius=1000)

        # --- C. Dinner (Near Last Place) ---
        last_key = sorted_keys[-1]
        last_place = timeline[last_key]
        
        if last_place['id'] != "LUNCH_BREAK":
            print(f"  > Dinner near: {last_place['name']}")
            restaurant_data["dinner"] = get_nearby_restaurants(
                last_place['latitude'], 
                last_place['longitude'], 
                preference=meal_type, 
                radius=1000
            )

    # 3. Construct Final Object
    final_schedule_json = {
        "timeline": timeline,        # The places & times
        "restaurants": restaurant_data # The separated dining options
    }

    # 7. Calculate Unvisited
    all_place_ids = set([p['id'] for p in places_json])
    unvisited_list = list(all_place_ids - visited_ids)

    return final_schedule_json, unvisited_list

# ==========================================
# 3. EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Test Data
    places_data =[
  {
    "name": "Sri Chamarajendra Park",
    "category": "Mandatory",
    "latitude": 12.9779291,
    "longitude": 77.5951549,
    "time_duration_hours": 1.5,
    "smart_score": 1000022.66,
    "price": 20,
    "schedule": {
      "friday": "6 AM-6 PM",
      "monday": "Closed",
      "saturday": "6 AM-6 PM",
      "sunday": "6 AM-6 PM",
      "thursday": "6 AM-6 PM",
      "tuesday": "6 AM-6 PM",
      "wednesday": "6 AM-6 PM"
    },
    "id": 1
  },
  {
    "name": "Visvesvaraya Industrial & Technological Museum",
    "category": "Museums",
    "latitude": 12.9751593,
    "longitude": 77.5964198,
    "time_duration_hours": 2.5,
    "smart_score": 34.37,
    "price": 40,
    "schedule": {
      "friday": "9:30 AM-6 PM",
      "monday": "9:30 AM-6 PM",
      "saturday": "9:30 AM-6 PM",
      "sunday": "9:30 AM-6 PM",
      "thursday": "9:30 AM-6 PM",
      "tuesday": "9:30 AM-6 PM",
      "wednesday": "9:30 AM-6 PM"
    },
    "id": 2
  },
  {
    "name": "Gurudwara Sri Guru Singh Sabha",
    "category": "Spiritual",
    "latitude": 12.9761875,
    "longitude": 77.6204375,
    "time_duration_hours": 1.5,
    "smart_score": 33.21,
    "price": 0,
    "schedule": {
      "friday": "3:30 AM-8:30 PM",
      "monday": "3:30 AM-8:30 PM",
      "saturday": "3:30 AM-8:30 PM",
      "sunday": "3:30 AM-8:30 PM",
      "thursday": "3:30 AM-8:30 PM",
      "tuesday": "3:30 AM-8:30 PM",
      "wednesday": "3:30 AM-8:30 PM"
    },
    "id": 3
  },
  {
    "name": "Lalbagh Botanical Garden",
    "category": "Parks",
    "latitude": 12.9494158,
    "longitude": 77.5846805,
    "time_duration_hours": 1.0,
    "smart_score": 23.04,
    "price": 20,
    "schedule": {
      "friday": "5 AM-7 PM",
      "monday": "5 AM-7 PM",
      "saturday": "5 AM-7 PM",
      "sunday": "5 AM-7 PM",
      "thursday": "5 AM-7 PM",
      "tuesday": "5 AM-7 PM",
      "wednesday": "5 AM-7 PM"
    },
    "id": 4
  },
  {
    "name": "Bugle Rock Park",
    "category": "Parks",
    "latitude": 12.9427052,
    "longitude": 77.5694287,
    "time_duration_hours": 1.0,
    "smart_score": 19.83,
    "price": 20,
    "schedule": {
      "friday": "5:30-10 AM, 4-8:30 PM",
      "monday": "5:30-10 AM, 4-8:30 PM",
      "saturday": "5:30-10 AM, 4-8:30 PM",
      "sunday": "5:30-10 AM, 4-8:30 PM",
      "thursday": "5:30-10 AM, 4-8:30 PM",
      "tuesday": "5:30-10 AM, 4-8:30 PM",
      "wednesday": "5:30-10 AM, 4-8:30 PM"
    },
    "id": 5
  },
  {
    "name": "Udupi Garden Park",
    "category": "Parks",
    "latitude": 12.9177592,
    "longitude": 77.6104461,
    "time_duration_hours": 1.0,
    "smart_score": 17.03,
    "price": 20,
    "schedule": {
      "friday": "5-11 AM, 4-8 PM",
      "monday": "5 AM-11 PM",
      "saturday": "5-11 AM, 4-8 PM",
      "sunday": "5-11 AM, 4-8 PM",
      "thursday": "5-11 AM, 4-8 PM",
      "tuesday": "5-11 AM, 4-8 PM",
      "wednesday": "5-11 AM, 4-8 PM"
    },
    "id": 8
  }
]

    # --- CALLING THE FUNCTION WITH NEW PARAMS ---
    # Example: 'slow' pace (longer visits) and a strict budget of 500
    itinerary, unvisited = generate_itinerary(
        places_json=places_data,
        hotel_name="Grand Hotel",
        hotel_lat=12.9716,
        hotel_lon=77.5946,
        day="friday",
        daily_budget=2500,  # Strict budget (Fateh Sagar Lake cost 1000, so it should be skipped)
        pace="Relaxed"        # Slow pace (1.5h becomes 1.875h)
    )
    j=json.dumps(itinerary)
    l=[j,unvisited]
    
    print(l)
    