import requests
import random
from typing import List, Dict, Any, Optional
import datetime # Added for date validation
import os
from dotenv import load_dotenv

load_dotenv()   # loads .env into environment variables

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# --- Amadeus API Credentials and Endpoints ---
# NOTE: These are test credentials provided by Amadeus and may require 
# occasional refreshing or replacement.


AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
HOTELS_BY_GEOCODE_URL = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-geocode"
HOTEL_OFFERS_URL = "https://test.api.amadeus.com/v3/shopping/hotel-offers"

# Rough chain tiers for estimation
LUXURY_CHAINS = {"MC", "RZ", "OB", "TJ", "DH"}
MID_CHAINS = {"BW", "UI", "WV", "SX", "AA"}
BUDGET_CHAINS = {"ON", "YX", "UZ", "IL", "HS", "VP"}


def get_access_token() -> str:
    """Get OAuth access token from Amadeus."""
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    try:
        response = requests.post(
            AUTH_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching token. Check CLIENT_ID/SECRET. Response: {e}")
        raise


def estimate_price(chain_code: str, distance_km: Optional[float]) -> int:
    """
    Estimate price per night in INR using rough heuristics.
    """
    chain_code = (chain_code or "").upper()
    base: float

    if chain_code in LUXURY_CHAINS:
        base = 12000.0
    elif chain_code in MID_CHAINS:
        base = 7000.0
    elif chain_code in BUDGET_CHAINS:
        base = 3500.0
    else:
        base = 5000.0

    if distance_km is not None:
        if distance_km > 5:
            base *= 0.8
        elif distance_km > 3:
            base *= 0.9

    return int(base)


def get_hotels_near(
    lat: float,
    lon: float,
    check_in: str,
    check_out: str,
    radius_km: int = 5,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Finds hotels near a geographic location, fetches real or estimated prices,
    and returns a filtered list of key hotel details.
    """
    try:
        # Validate dates early
        datetime.datetime.strptime(check_in, '%Y-%m-%d')
        datetime.datetime.strptime(check_out, '%Y-%m-%d')
        if check_in >= check_out:
            raise ValueError("Check-in date must be before Check-out date.")
    except ValueError as e:
        print(f"\n❌ Date Error: {e}. Dates must be in YYYY-MM-DD format.")
        return []

    try:
        token = get_access_token()
    except Exception:
        return [] # Stop if token fails

    # ---------- Step 1: Hotels by geocode ---------- #
    params = {
        "latitude": lat,
        "longitude": lon,
        "radius": radius_km,
        "radiusUnit": "KM",
    }

    try:
        resp = requests.get(
            HOTELS_BY_GEOCODE_URL,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        resp.raise_for_status() # Raises for 4XX/5XX errors
        hotels = resp.json().get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"\n❌ API Error in Step 1 (by-geocode). Check Lat/Lon/Radius. Details: {e}")
        return []

    # Sort by distance and keep top N
    hotels = sorted(hotels, key=lambda h: h.get("distance", {}).get("value", 9999))
    hotels = hotels[:limit]
    hotel_ids = [h["hotelId"] for h in hotels if "hotelId" in h]

    # ---------- Step 2: Real offers (if available) ---------- #
    offers_by_id: Dict[str, Dict[str, str]] = {}

    if hotel_ids:
        try:
            offers_resp = requests.get(
                HOTEL_OFFERS_URL,
                params={
                    "hotelIds": ",".join(hotel_ids),
                    "checkInDate": check_in,
                    "checkOutDate": check_out,
                    "adults": 2,
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
            offers_resp.raise_for_status() 
            offers_data = offers_resp.json().get("data", [])

            for item in offers_data:
                hid = item.get("hotel", {}).get("hotelId")
                offers = item.get("offers", [])
                if not hid or not offers:
                    continue
                first = offers[0]
                price = first.get("price", {})
                offers_by_id[hid] = {
                    "total": price.get("total", "N/A"),
                    "currency": price.get("currency", "INR"),
                }
        except requests.exceptions.RequestException as e:
            # Note: This is common if no offers exist, but we proceed with estimation
            print(f"⚠️ Warning: Could not fetch real offers (hotel-offers). Proceeding with estimates. Details: {e}")


    # ---------- Step 3: Combine + filter final output ---------- #
    results: List[Dict[str, Any]] = []
    for h in hotels:
        hid = h.get("hotelId")
        name = h.get("name")
        distance_km = h.get("distance", {}).get("value")
        chain = h.get("chainCode")

        # Extract address, lat, and long here
        address_parts = h.get("address", {})
        full_address = ", ".join(
            filter(None, [
                address_parts.get("lines", [""])[0] if address_parts.get("lines") else None,
                address_parts.get("cityName"),
                address_parts.get("stateCode"),
                address_parts.get("postalCode"),
                address_parts.get("countryCode")
            ])
        )
        geo = h.get("geoCode", {})
        latitude = geo.get("latitude")
        longitude = geo.get("longitude")

        price: Any
        currency: str
        if hid in offers_by_id:
            # Real price from API
            price = offers_by_id[hid]["total"]
            currency = offers_by_id[hid]["currency"]
        else:
            # No offer -> estimate price
            est = estimate_price(chain, distance_km)
            noise_factor = random.uniform(0.9, 1.1)
            price = int(est * noise_factor)
            currency = "INR"

        # Final Output Filtering (The required format)
        results.append(
            {
                "hotel_name": name,
                "hotel_id": hid,
                "address": full_address,
                "latitude": latitude,
                "longitude": longitude,
                "price": price,
                "currency": currency,
                "distance_km": distance_km,
            }
        )

    return results


# ---------------- MODULE EXECUTION (for direct running) ---------------- #
if __name__ == "__main__":
    try:
        latitude = float(input("Enter Latitude: "))
        longitude = float(input("Enter Longitude: "))
        check_in = input("Enter Check-in Date (YYYY-MM-DD): ")
        check_out = input("Enter Check-out Date (YYYY-MM-DD): ")

        print("\nSearching for hotels...")
        hotels = get_hotels_near(latitude, longitude, check_in, check_out)
        
        if hotels:
            print("\n*** FINAL FORMATTED HOTEL RESULTS ***")
            for h in hotels:
                print("-" * 50)
                print(f"Name: {h['hotel_name']}")
                print(f"ID: {h['hotel_id']}")
                print(f"Address: {h['address']}")
                print(f"Latitude: {h['latitude']}")
                print(f"Longitude: {h['longitude']}")
                print(f"Price: {h['price']} {h['currency']}")
                print(f"Distance: {h['distance_km']:.2f} km")
        else:
            print("\nNo hotels found, or a critical error occurred during the search.")
            
    except ValueError:
        print("\nInvalid input. Please ensure latitude/longitude are numbers.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")