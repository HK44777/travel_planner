import os
import time
import json
import math
import numpy as np
import pandas as pd
from celery import Celery
from k_means_constrained import KMeansConstrained
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import poi_get  # Ensure this file is in the same directory
from dotenv import load_dotenv

load_dotenv() 

# --- 1. Configure Celery ---
# Ensure Redis is running on localhost:6379
celery = Celery(
    'tasks', 
    broker='redis://localhost:6379/0', 
    backend='redis://localhost:6379/0'
)

# --- 2. Define Data Models (for Gemini) ---
class PriceEstimate(BaseModel):
    cost_inr: float = Field(description="The estimated total daily cost in Indian Rupees (INR) as a numeric value.")

# --- 3. Helper Functions ---
def get_nearest_clusters(current_cluster_id, all_centers, top_n=2):
    """
    Calculates distances from the current cluster to all others 
    and returns the IDs of the nearest 'top_n' clusters.
    """
    distances = []
    current_center = all_centers[current_cluster_id]
    
    for other_id, other_center in enumerate(all_centers):
        if current_cluster_id == other_id:
            continue
        
        # Calculate Euclidean distance
        dist = np.linalg.norm(current_center - other_center)
        distances.append((other_id, dist))
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])
    
    # Return just the IDs of the top N closest clusters
    return [d[0] for d in distances[:top_n]]

# --- 4. Define Tasks ---

@celery.task
def get_travel_cost(city_name: str, budget_type: str = "basic"):
    """
    Celery task to calculate travel cost using Google Gemini.
    """
    print(f"--- [Task Started] Fetching estimate for: {city_name} ({budget_type}) ---")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found in environment variables."}

    multipliers = {
        "basic": 1.0,
        "economy": 1.25,
        "standard": 1.5,
        "premium": 2.0,
        "luxury": 2.25
    }
    
    budget_key = budget_type.lower().strip()
    multiplier = multipliers.get(budget_key, 1.0)

    parser = PydanticOutputParser(pydantic_object=PriceEstimate)
    
    template = """
    You are a travel expert. Calculate the daily BASE cost (Basic Budget) for a traveler in {city}.
    Include: 3 Meals, Local Transport, and Entry Fee for some tourist places.
    Give me correct logical price
    Calculate the total cost in INDIAN RUPEES (INR).
    
    IMPORTANT: You must return strictly valid JSON matching the format below.
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["city"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.7, 
            google_api_key=api_key
        )

        chain = prompt | llm | parser
        prices = []

        for i in range(3):
            try:
                result = chain.invoke({"city": city_name})
                prices.append(result.cost_inr)
            except Exception:
                continue

        if prices:
            base_avg = sum(prices) / len(prices)
            final_adjusted_cost = base_avg * multiplier
            
            result_data = {
                "city": city_name,
                "budget_type": budget_type,
                "base_avg": round(base_avg, 2),
                "final_cost": round(final_adjusted_cost, 2),
                "currency": "INR"
            }
            return result_data
        else:
            return {"error": "Failed to retrieve prices from Gemini API."}

    except Exception as e:
        return {"error": str(e)}


@celery.task
def perform_clustering(destination: str, days: int, mandatory_places: list, preferences: list, dislikes: list):
    """
    Celery task that fetches POI data using `poi_get` and performs K-Means Constrained clustering.
    
    Args:
        destination (str): City name (e.g., "Udaipur")
        days (int): Number of days for the trip (Number of Clusters)
        mandatory_places (list): List of mandatory place names
        preferences (list): List of user preferences (e.g., ["history"])
        dislikes (list): List of user dislikes
        
    Returns:
        dict: Final structured JSON containing Places, Centroid, and Clustering info.
    """
    print(f"--- [Task Started] Clustering for {destination} ({days} days) ---")

    try:
        # 1. Get data from external source
        # This calls your existing poi_get.py logic
        input_data = poi_get.run_travel_planner_external(
            destination,
            days,
            mandatory_places,
            preferences,
            dislikes
        )

        # 2. Data Preparation
        if not input_data or 'places' not in input_data:
             return {"error": "No places data returned from poi_get."}

        df = pd.DataFrame(input_data['places'])
        
        if df.empty:
            return {"error": "Places DataFrame is empty."}

        total_places = len(df)
        
        # Ensure required columns exist
        required_cols = ['score', 'is_mandatory', 'latitude', 'longitude']
        for col in required_cols:
            if col not in df.columns:
                if col == 'is_mandatory': df[col] = False
                elif col == 'latitude': df[col] = 0.0
                elif col == 'longitude': df[col] = 0.0

        # Extract Features for Clustering
        X = df[['latitude', 'longitude']]
        
        # 3. Dynamic Constraint Calculation
        K = days
        avg_places_per_day = total_places // K
        
        # Handle edge case where there are fewer places than days
        if avg_places_per_day < 1:
            avg_places_per_day = 1
            K = total_places # Reduce clusters to match place count if necessary

        size_min = max(1, int(avg_places_per_day * 0.4)) 
        size_max = math.ceil(avg_places_per_day * 0.6)

        # Safety buffer for constraints to prevent infeasible solutions
        if K * size_min > total_places:
            size_min = total_places // K
        if K * size_max < total_places:
            size_max = math.ceil(total_places / K) + 1

        # 4. Clustering
        clf = KMeansConstrained(
            n_clusters=K,
            size_min=size_min,
            size_max=size_max,
            random_state=42
        )

        df['cluster_label'] = clf.fit_predict(X)
        cluster_centers = clf.cluster_centers_

        # 5. Constructing the Output
        
        # A. Places Data
        # Convert to dict and handle NaN values if any by replacing with None for JSON safety
        places_output = df.where(pd.notnull(df), None).to_dict(orient='records')

        # B. Centroid Data
        cluster_center_values = {}
        for idx, center in enumerate(cluster_centers):
            # center[0] is lat, center[1] is long based on X structure
            cluster_center_values[str(idx)] = {
                "lat": float(center[0]),
                "long": float(center[1])
            }
        avg_center_lat = np.mean(cluster_centers[:, 0])
        avg_center_long = np.mean(cluster_centers[:, 1])
        centroid_output = {
            "values": cluster_center_values,
            "avg_lat": float(avg_center_lat),
            "avg_long": float(avg_center_long)
        }

        # C. Clustering Data
        clustering_output = {}
        grouped = df.groupby('cluster_label')

        for cluster_id, group_df in grouped:
            # Basic Lists
            place_ids = group_df['id'].tolist()
            
            # Metrics
            # Check if smart_score exists, else fallback to 'score' or 0
            score_col = 'smart_score' if 'smart_score' in group_df.columns else 'score'
            if score_col in group_df.columns:
                avg_score = round(float(group_df[score_col].mean()), 2)
            else:
                avg_score = 0.0
            
            mandatory_count = int(group_df['is_mandatory'].sum())
            
            # Nearest Clusters
            # nearest = get_nearest_clusters(int(cluster_id), cluster_centers, top_n=2)
            
            clustering_output[str(cluster_id)] = {
                "place_ids": place_ids,
                "stats": {
                    "avg_score": avg_score,
                    "mandatory_count": mandatory_count
                }
            }

        final_structure = {
            "Places": places_output,
            "centroid": centroid_output,
            "clustering": clustering_output
        }
        
        print(f"--- [Task Finished] Clustering completed for {len(df)} places. ---")
        return final_structure

    except Exception as e:
        print(f"--- [Task Failed] Error: {e}")
        return {"error": str(e)}
