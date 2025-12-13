import requests
import os
import time
import math
import json
from statistics import mode, StatisticsError
from dotenv import load_dotenv

# Load API key
# IMPORTANT: Ensure your SERPAPI_KEY is set in your environment or a .env file.
# Note: The SERPAPI_KEY is hardcoded here for completion but should ideally be loaded from environment variables.
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# ==========================================
# 1. CONFIGURATION
# ==========================================

ALL_PREFERENCES = {
    "history": { "query": "top historical forts palaces monuments {city}", "types": ["historical_landmark", "tourist_attraction", "fort"], "icon": "🏛" },
    "spiritual": { "query": "famous temples churches mosques gurudwaras {city}", "types": ["hindu_temple", "church", "mosque", "place_of_worship"], "icon": "🙏" },
    "museums": { "query": "best museums art galleries science centers {city}", "types": ["museum", "art_gallery"], "icon": "🎨" },
    "parks": { "query": "beautiful city parks botanical gardens {city}", "types": ["park", "botanical_garden"], "icon": "🌳" },
    "wildlife": { "query": "zoo wildlife sanctuary bird sanctuary aquarium {city}", "types": ["zoo", "aquarium", "park"], "icon": "🦁" },
    "lakes": { "query": "famous lakes river fronts waterfalls {city}", "types": ["tourist_attraction", "park", "natural_feature"], "icon": "🌊" },
    "adventure": { "query": "amusement parks water parks theme parks {city}", "types": ["amusement_park", "tourist_attraction"], "icon": "🎢" },
    "malls": { "query": "biggest shopping malls luxury malls {city}", "types": ["shopping_mall"], "icon": "🛍" },
    "markets": { "query": "famous street markets bazaars flea markets {city}", "types": ["market", "shopping_mall", "tourist_attraction"], "icon": "🎪" },
    "entertainment": { "query": "movie theaters bowling gaming zones {city}", "types": ["movie_theater", "bowling_alley", "casino"], "icon": "🎬" },
    "trekking": { "query": "hiking trails nature trails trekking spots {city}", "types": ["park", "tourist_attraction"], "icon": "🧗" },
    "sports": { "query": "stadiums sports complexes cricket grounds {city}", "types": ["stadium", "gym"], "icon": "🏟" }
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def format_time_str(hour_24):
    """Converts 24-hour time (0-23) to 12-hour AM/PM string."""
    if hour_24 >= 24: hour_24 = 0
    if hour_24 == 0: return "12 AM"
    elif hour_24 < 12: return f"{hour_24} AM"
    elif hour_24 == 12: return "12 PM"
    else: return f"{hour_24 - 12} PM"

def clean_schedule_text(schedule_data):
    """
    Cleans Unicode characters (\u202f, \u2013) from the schedule data for readability.
    """
    if isinstance(schedule_data, dict):
        cleaned_schedule = {}
        for day, timing in schedule_data.items():
            if isinstance(timing, str):
                cleaned_schedule[day] = timing.replace('\u202f', ' ').replace('\u2013', '-')
            else:
                cleaned_schedule[day] = timing
        return cleaned_schedule
    elif isinstance(schedule_data, str):
        return schedule_data.replace('\u202f', ' ').replace('\u2013', '-')
    return schedule_data

def get_detailed_hours(place):
    """Extracts or estimates opening/closing hours using multiple strategies."""
    
    # Strategy 1: Operating Hours (Most Detailed)
    operating_hours = place.get('operating_hours')
    if operating_hours:
        schedule_lines = {day: operating_hours.get(day, "Closed") for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']}
        # We only need the full_schedule, so we skip generating a summary string here.
        return {"full_schedule": schedule_lines}

    # Strategy 2: Graph Mode (Popular Times Estimation)
    popular_times = place.get('popular_times', [])
    if popular_times:
        start_hours = []
        end_hours = []
        for day in popular_times:
            data = day.get('data', [])
            if not data: continue
            first_idx = next((i for i, x in enumerate(data) if x > 0), None)
            last_idx = next((i for i, x in enumerate(reversed(data)) if x > 0), None)
            
            if first_idx is not None and last_idx is not None:
                start_hours.append(first_idx)
                end_hours.append((23 - last_idx) + 1)

        if start_hours and end_hours:
            try:
                common_start = mode(start_hours)
                common_end = mode(end_hours)
                
                # Format the estimated schedule for the JSON output
                if common_start == 0 and common_end >= 24:
                    schedule_text = "Open 24 Hours (Estimated)"
                else:
                    schedule_text = f"Opens at {format_time_str(common_start)} and Closes at {format_time_str(common_end)} (Estimated)"
                
                return {"full_schedule": schedule_text}
            except StatisticsError:
                pass 

    # Strategy 3: Text Parsing (Fallback)
    text = place.get('open_state', '')
    if not text: return {"full_schedule": "Hours not available"}
    
    return {"full_schedule": text}

def estimate_native_price(place_name, category):
    name = place_name.lower().strip()
    cat = category.lower().strip()

    # Religious & Public = Free
    if any(k in name for k in ['temple', 'mandir', 'gurudwara', 'mosque', 'church', 'mall', 'market', 'street', 'beach']):
        return 0

    # Nature/Trekking
    if 'trekking' in cat or 'trail' in name or 'hills' in name:
        return "Free (or nominal ₹50 fee)"

    if 'safari' in name or 'national park' in name: return 400
    if 'zoo' in name: return "Approx. ₹100"
    if 'adventure' in cat or 'wonderla' in name: return 1200
    if any(k in name for k in ['fort', 'palace', 'museum']): return 40
    if any(k in name for k in ['park', 'garden', 'lake']): return 20
    
    return 0

def estimate_time_to_spend(place_name, category):
    """
    Estimates time needed based on category and name and returns a usable float (in hours).
    
    Returns: float (e.g., 2.5 for 2-3 hours)
    """
    cat = category.lower()
    name = place_name.lower()
    
    # Logic to return the midpoint or a sensible number in hours (float)
    if any(k in cat for k in ["zoo", "wildlife", "adventure"]): return 3.5  # 3-4 hours
    if any(k in cat for k in ["museum", "history", "fort", "malls", "trekking", "entertainment"]): return 2.5  # 2-3 hours
    if any(k in cat for k in ["lakes", "parks"]): return 1.0  # 45 mins - 1 hour
    return 1.5 # Default 1-2 hours


def calculate_smart_score(place, is_mandatory, is_hidden_gem):
    """Calculates the weighted score for prioritization."""
    rating = place.get('rating', 0)
    reviews = place.get('reviews', 0)
    # Score = Rating * log10(Reviews + 1)
    popularity_score = rating * math.log10(reviews + 1) if reviews > 0 else 0
    
    boost = 0
    # Mandatory places get a huge score boost to ensure they are always at the top
    if is_mandatory: boost = 1000000 
    elif is_hidden_gem: boost = 15
    return round(popularity_score + boost, 2)

# ==========================================
# 3. SEARCH FUNCTIONS
# ==========================================

def search_mandatory_place(place_name, city):
    """Searches for a specific place the user MUST visit."""
    clean_name = place_name.strip()
    queries = [f"{clean_name} {city}", f"{clean_name}"]
    
    for q in queries:
        url = "https://serpapi.com/search"
        params = {'engine': 'google_maps', 'q': q, 'api_key': SERPAPI_KEY, 'num': 1} 
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status() 
            results = response.json().get('local_results', [])
            if results:
                place = results[0].copy()
                place['is_mandatory'] = True
                place['preference_category'] = 'Mandatory'
                return place
        except Exception:
            continue
            
    # Return a default object if the search fails
    return {
        'title': clean_name, 'address': f"Location in {city} (Search Failed)",
        'rating': 0, 'reviews': 0, 'gps_coordinates': {'latitude': 0, 'longitude': 0},
        'is_mandatory': True, 'preference_category': 'Mandatory'
    }

def get_places_by_category(category_key, city):
    """Searches for places based on a preference category."""
    if not SERPAPI_KEY:
        return []
        
    config = ALL_PREFERENCES[category_key]
    url = "https://serpapi.com/search"
    params = {'engine': 'google_maps', 'q': config["query"].format(city=city), 'api_key': SERPAPI_KEY, 'num': 8}
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        results = response.json().get('local_results', [])
        for p in results:
            p['preference_category'] = category_key
            p['is_mandatory'] = False
            p['is_hidden_gem'] = False
        return results
    except Exception:
        return []

# ==========================================
# 4. FINAL PROCESSORS
# ==========================================

def process_final_list(places, disliked_categories):
    """Deduplicates, filters, enriches data, and finalizes the list into the requested clean structured format."""
    unique_map = {}
    
    # Sort initially to ensure mandatory places are processed first for proper deduplication
    sorted_places = sorted(places, key=lambda x: x.get('is_mandatory', False), reverse=True)
    
    for p in sorted_places:
        key = p.get('title', '').lower().strip()
        if key and key not in unique_map: 
            unique_map[key] = p
            
    final_list_unsorted = []
    
    for p in unique_map.values():
        is_mandatory = p.get('is_mandatory', False)
        category = p.get('preference_category', '').lower()
        
        # Filter: Skip non-mandatory places whose category is disliked
        if not is_mandatory and category in disliked_categories:
            continue
            
        # Enrich Data
        # 1. Score calculation
        is_hidden_gem = p.get('is_hidden_gem', False)
        p['smart_score'] = calculate_smart_score(p, is_mandatory, is_hidden_gem)
        
        # 2. Time estimation (returns float)
        time_hours = estimate_time_to_spend(p.get('title', ''), category)
        
        p['native_price'] = estimate_native_price(p.get('title', ''), category)
        # 3. Schedule details
        hours_data = get_detailed_hours(p)
        cleaned_schedule = clean_schedule_text(hours_data['full_schedule'])
        if cleaned_schedule=="Temporarily closed":
            cleaned_schedule={
                "monday": "9 AM-9 PM",
                "tuesday": "9 AM-9 PM",
                "wednesday": "9 AM-9 PM",
                "thursday": "9 AM-9 PM",
                "friday": "9 AM-9 PM",
                "saturday": "9 AM-9 PM",
                "sunday": "9 AM-9 PM"
            }
        if cleaned_schedule=="Hours not available":
            cleaned_schedule={
                "monday": "9 AM-9 PM",
                "tuesday": "9 AM-9 PM",
                "wednesday": "9 AM-9 PM",
                "thursday": "9 AM-9 PM",
                "friday": "9 AM-9 PM",
                "saturday": "9 AM-9 PM",
                "sunday": "9 AM-9 PM"
            }
        if cleaned_schedule=="Open now":
            cleaned_schedule={
                "monday": "9 AM-9 PM",
                "tuesday": "9 AM-9 PM",
                "wednesday": "9 AM-9 PM",
                "thursday": "9 AM-9 PM",
                "friday": "9 AM-9 PM",
                "saturday": "9 AM-9 PM",
                "sunday": "9 AM-9 PM"
            }
        if cleaned_schedule== {
                "monday": "Open 24 hours",
                "tuesday": "Open 24 hours",
                "wednesday": "Open 24 hours",
                "thursday": "Open 24 hours",
                "friday": "Open 24 hours",
                "saturday": "Open 24 hours",
                "sunday": "Open 24 hours"
            }:
            cleaned_schedule={
                "monday": "9 AM-9 PM",
                "tuesday": "9 AM-9 PM",
                "wednesday": "9 AM-9 PM",
                "thursday": "9 AM-9 PM",
                "friday": "9 AM-9 PM",
                "saturday": "9 AM-9 PM",
                "sunday": "9 AM-9 PM"
            }
        gps = p.get('gps_coordinates', {})
        
        # Create the FINAL, CLEAN output dictionary
        final_list_unsorted.append({
            'name': p.get('title', 'Unknown Place'),
            'category': category.capitalize(),
            'latitude': gps.get('latitude', 0),
            'longitude': gps.get('longitude', 0),
            'address': p.get('address', 'Address not available'),
            'time_duration_hours': time_hours, 
            'smart_score': p['smart_score'],
            'schedule': cleaned_schedule,
            'price':p["native_price"] ,
            'is_mandatory': is_mandatory,
        })
        
    # Sort the list by smart_score in descending order (highest score first)
    final_list_unsorted.sort(key=lambda x: x['smart_score'], reverse=True)
    
    # Add the final 'id' variable after sorting
    final_list_with_id = []
    for i, place in enumerate(final_list_unsorted, 1):
        place['id'] = i  # Add the 1-based index as 'id'
        final_list_with_id.append(place)
        
    return final_list_with_id

def generate_output_structure(places, city):
    """Wraps the final list in a metadata dictionary."""
    return {
        "city": city,
        "itinerary_generated_on": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_places": len(places),
        "places": places
    }

# ==========================================
# 5. EXTERNAL CALLABLE FUNCTION (The main export)
# ==========================================

def run_travel_planner_external(
    city_name: str, 
    num_of_days: int = 1, # New parameter added
    mandatory_list: list = None, 
    liked_categories: list = None, 
    disliked_categories: list = None
) -> dict:
    """
    Core function to run the travel planner logic with specific inputs, 
    limited to 10 * num_of_days places.
    
    Args:
        city_name (str): The city name (e.g., "Delhi").
        num_of_days (int): The duration of the trip. Limits the final output to 10 * num_of_days places.
        mandatory_list (list): List of required place names (e.g., ["Red Fort"]).
        liked_categories (list): List of preferred category keys (e.g., ["history", "museums"]).
        disliked_categories (list): Optional list of categories to explicitly exclude.
        
    Returns:
        dict: The final structured itinerary data with clean keys/values.
    """
    if not SERPAPI_KEY:
        raise ValueError("SERPAPI_KEY is not set. Cannot run travel planner.")
    if not city_name:
        raise ValueError("City name cannot be empty.")
    if not isinstance(num_of_days, int) or num_of_days < 1:
        raise ValueError("num_of_days must be an integer greater than 0.")
        
    # Clean and normalize inputs
    mandatory_list = [m.strip() for m in (mandatory_list or [])]
    liked_categories = [k.lower().strip() for k in (liked_categories or [])]
    disliked_categories = [k.lower().strip() for k in (disliked_categories or [])]
    
    raw_places = []
    keys = list(ALL_PREFERENCES.keys())
    
    # 1. Search Mandatory Places
    for name in mandatory_list:
        raw_places.append(search_mandatory_place(name, city_name))
        time.sleep(0.5)
        
    # 2. Fetch Selected Preferences
    for pref in liked_categories:
        if pref in ALL_PREFERENCES:
            raw_places.extend(get_places_by_category(pref, city_name))
            time.sleep(0.5)
        
    # 3. Fetch Hidden Gems 
    unselected_prefs = [k for k in keys if k not in liked_categories]
    gem_scan_prefs = [k for k in unselected_prefs if k not in disliked_categories]
    
    for pref in gem_scan_prefs[:3]: 
        res = get_places_by_category(pref, city_name)
        # Gem Filter: high rating (>= 4.6) AND high reviews (> 2000)
        gems = [p for p in res if p.get('rating', 0) >= 4.6 and p.get('reviews', 0) > 2000]
        for g in gems[:1]: 
            g['is_hidden_gem'] = True
            raw_places.append(g)
        time.sleep(0.5)

    # 4. Final Processing and Structuring (This sorts and adds 'id')
    final_itinerary_list = process_final_list(raw_places, disliked_categories)
    
    # 5. Limit the list based on num_of_days
    max_places = 5 * num_of_days
    
    # Since the list is already sorted by smart_score, slicing takes the "top" places
    limited_list = final_itinerary_list[:max_places]
    
    # 6. Generate final output structure
    return generate_output_structure(limited_list, city_name)

# ==========================================
# 6. DEMO BLOCK (Optional, for direct testing)
# ==========================================

if __name__ == "__main__":
    print("--- Running External API Demo ---")
    
    # --- Define Inputs for Testing ---
    TEST_CITY = "Udaipur"
    TEST_MANDATORY = ["City Palace, Udaipur", "Jagdish Temple"] # Added another mandatory place
    TEST_LIKES = ["history", "lakes"] # Removed 'spiritual' to see a different mix
    TEST_DISLIKES = ["sports", "entertainment"] 
    TEST_DAYS = 2  # Max places: 20
    # ---------------------------------
    
    try:
        print(f"Generating itinerary for {TEST_CITY} for {TEST_DAYS} days...")
        
        final_data = run_travel_planner_external(
            city_name=TEST_CITY,
            num_of_days=TEST_DAYS,
            mandatory_list=TEST_MANDATORY,
            liked_categories=TEST_LIKES,
            disliked_categories=TEST_DISLIKES
        )

        print("\n✅ ITINERARY GENERATED SUCCESSFULLY (JSON Output):")
        print("=" * 80)
        print(f"Expected Max Places: {10 * TEST_DAYS}. Actual Places: {final_data['total_places']}")
        print(json.dumps(final_data, indent=4))
        print("=" * 80)
        
        # Quick check for IDs
        if final_data['places']:
            print(f"First Place ID: {final_data['places'][0]['id']}")
            print(f"Last Place ID: {final_data['places'][-1]['id']}")


    except ValueError as e:
        print(f"CONFIGURATION ERROR: {e}")
    except Exception as e:
        print(f"AN UNEXPECTED ERROR OCCURRED: {e}")