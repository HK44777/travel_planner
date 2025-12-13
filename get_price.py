import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()   # loads .env into environment variables

google_key = os.getenv("GOOGLE_API_KEY")

# --- 1. Define Output Structure ---
class PriceEstimate(BaseModel):
    cost_inr: float = Field(description="The estimated total daily cost in Indian Rupees (INR) as a numeric value.")

def get_average_cost_gemini(city_name: str, budget_type: str = "basic"):
    # Define multipliers
    multipliers = {
        "basic": 1.0,
        "economy": 1.25,
        "standard": 1.5,
        "premium": 2.0,
        "luxury": 2.25
    }
    
    # Normalize input to lowercase and get multiplier (default to 1.0 if not found)
    budget_key = budget_type.lower().strip()
    multiplier = multipliers.get(budget_key, 1.0)
    
    print(f"--- Budget Type: {budget_type.capitalize()} (Multiplier: x{multiplier}) ---")

    # Setup the parser
    parser = PydanticOutputParser(pydantic_object=PriceEstimate)
    
    # Define the Prompt
    # We ask for a 'Base' cost in the prompt, which we will later multiply
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

    # --- 2. Initialize Model (Gemini) ---
    # NOTE: "gemini-2.5-flash" doesn't exist yet publicly; using "gemini-1.5-flash"
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.7, 
        google_api_key=google_key # Best practice: Use env variable
    )

    # --- 3. Create Chain ---
    chain = prompt | llm | parser

    print(f"--- Fetching base INR prices for {city_name} (3 Attempts) ---")
    prices = []

    # Loop 3 times to get an average from the SAME model
    for i in range(3):
        try:
            print(f"   Request {i+1}...", end=" ")
            result = chain.invoke({"city": city_name})
            print(f"Got Base: ₹{result.cost_inr}")
            prices.append(result.cost_inr)
        except Exception as e:
            print(f"Failed: {e}")

    # --- 4. Calculate Average & Apply Multiplier ---
    if prices:
        base_avg = sum(prices) / len(prices)
        final_adjusted_cost = base_avg * multiplier
        
        print(f"   > Base Average: ₹{base_avg:.2f}")
        print(f"   > Applying Multiplier x{multiplier}")
        
        return round(final_adjusted_cost, 2)
        
    return "Error"

# --- Run ---
if __name__ == "__main__":
    # Set your key here safely or ensure it is in your environment variables
    # os.environ["GOOGLE_API_KEY"] = "YOUR_ACTUAL_KEY_HERE"
    
    # 1. Ask for inputs
    city = input("Enter City Name: ")
    b_type = input("Enter Budget Type (Basic, Economy, Standard, Premium, Luxury): ")
    
    # 2. Get Cost
    final_avg = get_average_cost_gemini(city, b_type)
        
    print("-" * 30)
    print(f"FINAL ESTIMATED DAILY COST ({b_type.upper()}): ₹{final_avg}")