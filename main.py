import json
import os
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from pydantic import BaseModel, EmailStr, ValidationError, field_validator, model_validator, PositiveInt 
from datetime import date
from typing import List, Optional, Literal 
from tasks import perform_clustering, get_travel_cost
from celery.result import AsyncResult
import tasks # Import the module to access the celery instance config if needed
import pandas as pd
import numpy as np
from datetime import datetime
# CHANGED: Removed FlaskForm, wtforms, and CSRFProtect imports

t={}
t_r={}


bus=pd.read_excel('Bus_Timings_dataset.xlsx')
flight=pd.read_excel('flight_data.xlsx')
train=pd.read_excel('Train_Timings_dataset.xlsx')

app = Flask(__name__)
# You MUST change this secret key in production!
# It's required for Flask-Login sessions
from dotenv import load_dotenv

load_dotenv() 
app.config["SECRET_KEY"] =os.getenv("SECRET_KEY")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SQLALCHEMY_DATABASE_URI")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# --- INITIALIZE EXTENSIONS ---
db = SQLAlchemy(app)

# This is critical for React:
#   - CORS(app): Allows all origins (e.g., localhost:3000)
#   - supports_credentials=True: Allows Flask to send the session cookie
#     to your React app.
CORS(app, supports_credentials=True)

login_manager = LoginManager()
login_manager.init_app(app)


# --- DATABASE MODEL (UserMixin is required by Flask-Login) ---


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, password):
        """Hashes and sets the password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Checks a given password against the stored hash."""
        return check_password_hash(self.password_hash, password)


class Trip(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # --- Location & Time ---
    origin_city = db.Column(db.String(100), nullable=False)
    destination_city = db.Column(db.String(100), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)

    # --- Budget (Separate Columns) ---
    budget_type = db.Column(db.String(50), nullable=True)  # e.g., 'Budget-Friendly', 'Luxury'
    budget_amount = db.Column(db.Integer, nullable=True)  # e.g., 50000

    # --- Group & Preferences ---
    num_people = db.Column(db.Integer, nullable=False)

    # Stored as JSON strings in TEXT columns
    preferences = db.Column(db.Text, nullable=True)  # '["museums", "parks"]'
    dislikes = db.Column(db.Text, nullable=True) 
    mandatory_places = db.Column(db.Text, nullable=True)  # '["India Gate", "Red Fort"]'

    # --- Other Constraints ---
    food_preference = db.Column(db.String(50), nullable=True)  # 'Veg-Only', 'Any'
    pace = db.Column(db.String(50), nullable=True)  # 'Relaxed', 'Moderate'
    needs_accommodation = db.Column(db.Boolean, nullable=True)
    needs_transport = db.Column(db.Boolean, nullable=True)
    transport_type = db.Column(db.String(50), nullable=True)
    accommodation_type = db.Column(db.String(50), nullable=True)
    acc_loc=db.Column(db.String(100), nullable=True)
    up_mode=db.Column(db.String(100), nullable=True)
    up_mode_id=db.Column(db.String(100), nullable=True)
    down_mode=db.Column(db.String(100), nullable=True)
    down_mode_id=db.Column(db.String(100), nullable=True)
    hotel_name=db.Column(db.String(100), nullable=True)
    hotel_address=db.Column(db.String(100), nullable=True)
    hotel_lat = db.Column(db.Float, nullable=True)
    hotel_long = db.Column(db.Float, nullable=True)
    hotel_price = db.Column(db.Float, nullable=True)


class Itinerary(db.Model):
    __tablename__ = 'itinerary'

    id = db.Column(db.Integer, primary_key=True)
    # Link it to your existing Trip table
    trip_id = db.Column(db.Integer, db.ForeignKey('trip.id', ondelete='CASCADE'), unique=True, nullable=False)
    # Store the entire structure here
    daily_plan = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    trip = db.relationship(
        'Trip', 
        backref=db.backref('itinerary', uselist=False, cascade="all, delete-orphan")
    )
# --- FLASK-LOGIN CALLBACKS ---

# This callback is used to reload the user object from the user ID
# stored in the session.
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# IMPORTANT: This handles what happens when a user who is NOT logged in
# tries to access a @login_required route.
# For an API, we return a JSON error, not redirect.
@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({"errors": {"auth": "You are not authorized. Please log in."}}), 401


# --- VALIDATION MODELS (using Pydantic) ---
# CHANGED: Replaced Flask-WTF forms with Pydantic models

class SignUpModel(BaseModel):
    name: str
    email: EmailStr  # Pydantic's built-in email validator
    password: str

    # Custom validator for password length
    @field_validator('password')
    @classmethod
    def password_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        return v


class LoginModel(BaseModel):
    email: EmailStr
    password: str

class TripModel(BaseModel):
    # Location & Time
    origin_city: str
    destination_city: str
    start_date: date  # Pydantic will parse "YYYY-MM-DD" strings
    end_date: date

    # Budget
    # Use Literal to restrict values, Optional for fields that can be null
    budget_type: Optional[Literal["Basic", "Economy", "Standard", "Premium", "Luxury"]] = None
    budget_amount: Optional[PositiveInt] = None  # Ensures it's an integer > 0

    # Group & Preferences
    num_people: PositiveInt # Ensures it's an integer > 0
    
    # We expect lists of strings from the frontend
    preferences: List[str] 
    dislikes: List[str] 
    mandatory_places: List[str]

    # Other Constraints
    food_preference: Literal["Veg-Only", "Any"]
    pace: Literal["Relaxed", "Moderate", "Fast-Paced"]
     # --- NEW --- Advanced Validator
    # This checks that the end_date is not before the start_date
    @model_validator(mode='after')
    def check_dates(self) -> 'TripModel':
        if self.start_date and self.end_date:
            if self.end_date < self.start_date:
                # This will be sent to the frontend as an error
                raise ValueError("End date cannot be before start date.")
        return self

class TransModel(BaseModel):
    needs_transport: Optional[bool] = None
    transport_type: Optional[Literal["Basic", "Economy", "Standard", "Premium", "Luxury"]] = None

class AccModel(BaseModel):
    needs_accommodation: Optional[bool] = None
    accommodation_type: Optional[Literal["Basic", "Economy", "Standard", "Premium", "Luxury"]] = None
    acc_loc:Optional[str]= None


# --- HELPER FUNCTION TO FORMAT Pydantic ERRORS ---
# CHANGED: Added this function to make Pydantic's errors
# look just like Flask-WTF's `form.errors` for your React frontend.
def format_pydantic_errors(error: ValidationError):
    """Converts Pydantic's error list into a dict {field: [messages]}."""
    errors = {}
    for err in error.errors():
        # 'loc' is a tuple, e.g., ('email',)
        field = err["loc"][0] if err["loc"] else "__root__"
        message = err["msg"]
        
        if field not in errors:
            errors[field] = []
        errors[field].append(message)
    return errors


# --- API ENDPOINTS ---

# A simple "hello world" route to check if the server is up
@app.route("/")
def hello():
    return jsonify({"message": "Flask server is running!"})


@app.route("/signup", methods=["POST"])
# CHANGED: Removed @csrf.exempt (CSRF protection was removed)
def signup():
    # Get JSON data from the React request
    data = request.get_json()

    try:
        # CHANGED: Validate data using Pydantic's model_validate
        model = SignUpModel.model_validate(data)

        # Check if email already exists
        existing_user = User.query.filter_by(email=model.email).first()
        if existing_user:
            return jsonify({"errors": {"email": ["Email address already in use."]}}), 400

        # Create new user
        # CHANGED: Use model.name, model.email, etc.
        new_user = User(name=model.name, email=model.email)
        new_user.set_password(model.password)  # This hashes the password

        # Add to database
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)

        return jsonify({"message": f"User {new_user.name} created successfully!"}), 201

    except ValidationError as e:
        # CHANGED: Handle Pydantic validation errors
        # Return the errors in the same format as Flask-WTF
        return jsonify({"errors": format_pydantic_errors(e)}), 400


@app.route("/login", methods=["POST"])
# CHANGED: Removed @csrf.exempt
def login():
    data = request.get_json()

    try:
        # CHANGED: Validate with Pydantic's LoginModel
        model = LoginModel.model_validate(data)

        # Find the user by email
        # CHANGED: Use model.email
        user = User.query.filter_by(email=model.email).first()

        # Check if user exists and if the password is correct
        # CHANGED: Use model.password
        if user and user.check_password(model.password):
            # *** THIS IS THE KEY FLASK-LOGIN FUNCTION ***
            # It creates the session and sends the 'Set-Cookie' header
            login_user(user)

            return jsonify(
                {
                    "message": f"Welcome back, {user.name}!",
                    "user": {"id": user.id, "name": user.name, "email": user.email},
                }
            ), 200
        else:
            # Failed login
            return jsonify({"errors": {"auth": ["Invalid email or password."]}}), 401

    except ValidationError as e:
        # CHANGED: Handle Pydantic validation errors
        # Return the errors in the same format as Flask-WTF
        return jsonify({"errors": format_pydantic_errors(e)}), 400


@app.route("/list_trips", methods=["GET"])
@login_required
def list_trips():
    today = date.today()
    
    # 1. Find all trips for the currently logged-in user
    trips = Trip.query.filter_by(user_id=current_user.id).all()
    
    trips_data = []
    
    # 2. Loop through them and format the data
    for trip in trips:
        # Calculate days left until the trip starts
        # If negative, the trip has already started
        days_left = (trip.start_date - today).days
        
        trips_data.append({
            "id": trip.id,
            # "user_id": trip.user_id,
            
            # Location & Time
            "origin_city": trip.origin_city,
            "destination_city": trip.destination_city,
            "start_date": trip.start_date.isoformat(), # Format as "YYYY-MM-DD" string
            "end_date": trip.end_date.isoformat(),
            # Calculated field
            "days_left": days_left
        })
        
    # 3. Return the list of trip objects
    return jsonify(trips_data), 200


@app.route("/delete_trip", methods=["POST"])
@login_required
def delete_trip_json():
    data = request.get_json()
    if not data or 'trip_id' not in data:
        return jsonify({"error": "Missing 'trip_id' in JSON body"}), 400

    try:
        trip_id = data['trip_id']
    except ValueError:
        return jsonify({"error": "Invalid trip ID format. Must be an integer."}), 400
    trip = Trip.query.filter_by(id=trip_id).first()
    print(trip)
    # 3. Handle case where the trip is not found or unauthorized
    if trip is None:
        return jsonify({"error": "Trip not found or unauthorized"}), 404

    try:
        # 4. Delete the trip
        db.session.delete(trip)
        
        # 5. Commit the transaction
        db.session.commit()
        
        # 6. Return success (200 OK with a confirmation message)
        return jsonify({"message": f"Trip ID {trip_id} successfully deleted."}), 200
        
    except Exception as e:
        # 7. Handle potential database errors
        db.session.rollback()
        print(f"Database error during trip deletion (POST): {e}")
        return jsonify({"error": "An internal error occurred during deletion"}), 500
    


# --- NEW --- Endpoint to create a new trip
@app.route("/create_trip", methods=["POST"])
@login_required  # <-- Ensures only a logged-in user can create a trip
def create_trip():
    data = request.get_json()

    try:
        # 1. Validate the incoming data using the Pydantic model
        model = TripModel.model_validate(data)

        # 2. Create the new Trip database object
        new_trip = Trip(
            # Get the user_id from the session (current_user)
            user_id=current_user.id, 

            # Map all the validated fields from the model
            origin_city=model.origin_city,
            destination_city=model.destination_city,
            start_date=model.start_date,
            end_date=model.end_date,
            budget_type=model.budget_type,
            budget_amount=model.budget_amount,
            num_people=model.num_people,

            # --- Convert lists to JSON strings ---
            # The database 'preferences' column is db.Text,
            # so we must store the list as a JSON string.

            preferences=json.dumps(model.preferences),
            dislikes=json.dumps(model.dislikes),
            mandatory_places=json.dumps(model.mandatory_places),
            food_preference=model.food_preference,  # ← ADD THIS
            pace=model.pace
        )
        if model.budget_type is not None:
            task1 = get_travel_cost.delay(model.destination_city, model.budget_type)
            t["get_travel_cost"]=task1.id
        d=model.end_date-model.start_date
        task2 = perform_clustering.delay(
        model.destination_city,
        int(d.days+1),
        model.mandatory_places,
        model.preferences,
        model.dislikes
    )
        t["perform_clustering"]=task2.id
        # 3. Add to the database
        db.session.add(new_trip)
        db.session.commit()

        # 4. Send a success response
        return jsonify({"message": "Trip created successfully!", "trip_id": new_trip.id,"tasks":t}), 201

    except ValidationError as e:
        # If validation fails, send back the formatted errors
        return jsonify({"errors": format_pydantic_errors(e)}), 400


@app.route("/update_transport", methods=["POST"])
@login_required
def update_transport():
    """
    Updates any/all optional details for an existing trip.
    This version manually checks for each expected key.
    """
    data = request.get_json()

    try:
        # 1. Validate the incoming data using the Pydantic model
        model = TransModel.model_validate(data)
        # 1. Get the trip_id from the raw JSON data
        trip_id = data.get('trip_id')
        if not trip_id:
            return jsonify({"errors": {"trip": "trip_id is missing from request."}}), 400

        # 2. Find the trip in the database
        trip = db.session.get(Trip, trip_id)

        # 3. Security Checks
        if not trip:
            return jsonify({"errors": {"trip": "Trip not found."}}), 404
        
        # ---
        # CRITICAL: You must keep this security check.
        # It ensures a user can only edit their *own* trips.
        # ---
        if 'needs_transport' in data:
            trip.needs_transport = data['needs_transport']
        if 'transport_type' in data:
            trip.transport_type = data['transport_type']
        db.session.commit()
        
        return jsonify({"message": "Trip details updated successfully!", "trip_id": trip.id}), 200
    
    except ValidationError as e:
        # If validation fails, send back the formatted errors
        return jsonify({"errors": format_pydantic_errors(e)}), 400

    except Exception as e:
        # Catch any other potential errors (like database errors)
        db.session.rollback()
        return jsonify({"errors": {"database": str(e)}}), 500
    

@app.route("/transport_option", methods=["POST"])
@login_required
def transport_option():
    try:
        data = request.get_json()
        trip_id = data.get('trip_id')
        trip = db.session.get(Trip, trip_id)

        if not trip:
            return jsonify({"error": "Trip not found"}), 404

        u_d = data.get('u_d')
        if u_d == 'u':
            src = trip.origin_city
            dest = trip.destination_city
            # Ensure start_date is a standard date object
            if isinstance(trip.start_date, str):
                start_date = pd.to_datetime(trip.start_date).date()
            else:
                start_date = trip.start_date
        else:
            src = trip.destination_city
            dest = trip.origin_city
            if isinstance(trip.end_date, str):
                start_date = pd.to_datetime(trip.end_date).date()
            else:
                start_date = trip.end_date

        bud_type = trip.transport_type
        ppl = trip.num_people

        # Helper function to calculate price limit safely
        def get_price_limit(transport_result, budget_type):
            if not transport_result:  # If list is empty, return Infinity (accept nothing or all)
                return 0
            
            prices = [item[0][1] for item in transport_result]
            if not prices: return 0

            budget_mapping = {'basic': 1, 'economy': 2, 'standard': 3, 'premium': 4, 'luxury': 5}
            multiplier = budget_mapping.get(budget_type.lower(), 3) # Default to standard
            
            min_price = min(prices)
            max_price = max(prices)
            price_difference = max_price - min_price
            return (multiplier / 5) * price_difference + min_price

        # --- PROCESS BUS ---
        bus['departure_date'] = pd.to_datetime(bus['departure_date'])
        
        filtered_buses = bus[
            (bus['source_city'] == src) & 
            (bus['destination_city'] == dest) & 
            (bus['departure_date'].dt.date == start_date) & 
            (bus['available_seats'] >= ppl)
        ].copy() # Use .copy() to avoid SettingWithCopy warnings

        # Logic to filter by price
        if not filtered_buses.empty:
            bus_result = []
            for index, row in filtered_buses.iterrows():
                bus_result.append([(row['bus_id'], row['price'])])
            
            limit = get_price_limit(bus_result, bud_type)
            # Filter the dataframe directly
            final_buses = filtered_buses[filtered_buses['price'] <= limit]
        else:
            final_buses = pd.DataFrame() # Empty if no buses found

        # --- PROCESS TRAIN ---
        train.columns = train.columns.str.strip()
        train['departure_date'] = pd.to_datetime(train['departure_date'])
        
        filtered_trains = train[
            (train['source_city'] == src) & 
            (train['destination_city'] == dest) & 
            (train['departure_date'].dt.date == start_date) & 
            (train['available_seats'] >= ppl)
        ].copy()

        if not filtered_trains.empty:
            train_result = []
            for index, row in filtered_trains.iterrows():
                train_result.append([(row['train_id'], row['price'])])
            
            limit = get_price_limit(train_result, bud_type)
            final_trains = filtered_trains[filtered_trains['price'] <= limit]
        else:
            final_trains = pd.DataFrame()

        # --- PROCESS FLIGHT ---
        flight['departure_date'] = pd.to_datetime(flight['departure_date'])
        
        filtered_flights = flight[
            (flight['source_city'] == src) & 
            (flight['destination_city'] == dest) & 
            (flight['departure_date'].dt.date == start_date) & 
            (flight['available_seats'] >= ppl)
        ].copy()

        if not filtered_flights.empty:
            flight_result = []
            for index, row in filtered_flights.iterrows():
                flight_result.append([(row['flight_id'], row['price'])])
            
            limit = get_price_limit(flight_result, bud_type)
            final_flights = filtered_flights[filtered_flights['price'] <= limit]
        else:
            final_flights = pd.DataFrame()

        # --- PREPARE RESPONSE ---
        # Convert Timestamps to strings to avoid JSON errors
        # We assume standard string conversion is fine for the response
        
        master_data = {
            "bus": final_buses.to_dict(orient='records'),
            "train": final_trains.to_dict(orient='records'),
            "flight": final_flights.to_dict(orient='records')
        }

        # FIX: Directly pass the dictionary to jsonify. 
        # Do NOT use json.dumps() here.
        return jsonify(master_data), 200

    except Exception as e:
        # This will print the actual error to your terminal so you can debug
        print(f"Error in transport_option: {e}")
        return jsonify({"error": str(e)}), 500
    



@app.route("/update_accomodation", methods=["POST"])
@login_required
def update_accomodation():
    """
    Updates any/all optional details for an existing trip.
    This version manually checks for each expected key.
    """
    data = request.get_json()

    try:
        model = AccModel.model_validate(data)
        # 1. Get the trip_id from the raw JSON data
        trip_id = data.get('trip_id')
        if not trip_id:
            return jsonify({"errors": {"trip": "trip_id is missing from request."}}), 400

        # 2. Find the trip in the database
        trip = db.session.get(Trip, trip_id)

        # 3. Security Checks
        if not trip:
            return jsonify({"errors": {"trip": "Trip not found."}}), 404
        
        # ---
        # CRITICAL: You must keep this security check.
        # It ensures a user can only edit their *own* trips.
        # ---
        if 'needs_accommodation' in data:
            trip.needs_accommodation = data['needs_accommodation']
        if 'accommodation_type' in data:
            trip.accommodation_type = data['accommodation_type']
        if 'acc_loc' in data:
            trip.acc_loc=data["acc_loc"]

        # 2. Prepare API request
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': trip.acc_loc,
                'format': 'json'
            }
            
            # Nominatim usage policy requires a custom User-Agent to identify the application
            # We need a unique User-Agent to avoid 403 Forbidden errors. 
            headers = {
                'User-Agent': 'FlaskGeocodingProject/1.0 (educational_test)' 
            }

            try:
                # 3. Send request to Nominatim API
                response = requests.get(url, params=params, headers=headers)
                response.raise_for_status() # Raise an exception for HTTP errors
                
                data = response.json()
                # 4. Handle empty results
                if not data:
                    return jsonify({'error': 'Location not found'}), 404

                # 5. Extract data from the first result
                first_result = data[0]
                trip.hotel_name=""
                trip.hotel_lat=first_result.get('lat')
                trip.hotel_long=first_result.get('lon')
                task_result = AsyncResult(t["perform_clustering"], app=tasks.celery)
                global t_r
                t_r=task_result.result
            except requests.RequestException as e:
                return jsonify({'error': f'Error connecting to Nominatim API: {str(e)}'}), 502
            
            except Exception as e:
                return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500


        db.session.commit()
        
        return jsonify({"message": "Trip details updated successfully!", "trip_id": trip.id}), 200

    except ValidationError as e:
        # If validation fails, send back the formatted errors
        return jsonify({"errors": format_pydantic_errors(e)}), 400

    except Exception as e:
        # Catch any other potential errors (like database errors)
        db.session.rollback()
        return jsonify({"errors": {"database": str(e)}}), 500


@app.route("/up_transport_choice",methods=["POST"])
@login_required
def up_transport_choice():
    data = request.get_json()
    trip_id = data.get('trip_id')
    if not trip_id:
        return jsonify({"errors": {"trip": "trip_id is missing from request."}}), 400

    # 2. Find the trip in the database
    trip = db.session.get(Trip, trip_id)
    trip.up_mode=data['up_mode']
    trip.up_mode_id=data['up_mode_id']
    db.session.commit()

    return jsonify({"message": "Choice updated successfully."}), 200

@app.route("/down_transport_choice",methods=["POST"])
@login_required
def down_transport_choice():
    data = request.get_json()
    trip_id = data.get('trip_id')
    if not trip_id:
        return jsonify({"errors": {"trip": "trip_id is missing from request."}}), 400

    # 2. Find the trip in the database
    trip = db.session.get(Trip, trip_id)
    trip.down_mode=data['down_mode']
    trip.down_mode_id=data['down_mode_id']
    db.session.commit()

    return jsonify({"message": "Choice updated successfully."}), 200


import requests
from typing import List, Dict, Any, Optional
import hotels
# from your_module import login_required 

# Helper function to calculate price limit safely (Defined outside the endpoint)
def get_price_limit_hotels(hotel_results: List[Dict[str, Any]], budget_type: str) -> float:
    """
    Calculates the maximum acceptable price based on min/max price and budget type.
    """
    if not hotel_results:
        return 0.0
    # print(hotel_results)
    prices = []
    for item in hotel_results:
        try:
            price = float(item.get('price')) 
            prices.append(price)
        except (ValueError, KeyError, TypeError):
            continue

    if not prices:
        return 0.0

    budget_mapping = {'basic': 1, 'economy': 2, 'standard': 3, 'premium': 4, 'luxury': 5}
    multiplier = budget_mapping.get(budget_type.lower(), 3)

    min_price = min(prices)
    # print(min_price)
    max_price = max(prices)
    # print(max_price)
    price_difference = max_price - min_price
    f_p=(multiplier / 5) * price_difference + min_price
    # print(f_p)
    return f_p


@app.route("/hotel_option", methods=["POST"])
# @login_required # Uncomment if you use this decorator
def hotel_option():
    try:
        data = request.get_json()
        
        trip_id = data.get('trip_id')
        trip = db.session.get(Trip, trip_id)
        task_result = AsyncResult(t["perform_clustering"], app=tasks.celery)
        
        # city = trip.origin_city
        # --- 1. Extract Input ---......HERE CELERY TASK RESULT I SDH GET
        global t_r
        t_r=task_result.result
        CITY_CENTERS = {
            "mumbai": {"lat": 19.0760, "lon": 72.8777},
            "bengaluru": {"lat": 12.9716, "lon": 77.5946},
            "hyderabad": {"lat": 17.3850, "lon": 78.4867}
        }

        # Normalize the city name to handle case sensitivity/spaces
        origin_city_key = trip.destination_city.strip().lower() if trip.origin_city else ""

        # --- 2. Determine Lat/Lon ---
        if origin_city_key in CITY_CENTERS:
            # Use precise hardcoded values for specific cities
            lat = CITY_CENTERS[origin_city_key]["lat"]
            lon = CITY_CENTERS[origin_city_key]["lon"]
        else:
            # Fallback to the original centroid logic for other cities
            lat = t_r["centroid"]["avg_lat"]
            lon = t_r["centroid"]["avg_long"]
        check_in = trip.start_date.strftime('%Y-%m-%d')
        check_out = trip.end_date.strftime('%Y-%m-%d')
        bud_type = trip.accommodation_type
        
        if not all([lat, lon, check_in, check_out, bud_type]):
            return jsonify({"error": "Missing required parameters.","t":t_r}), 400

        # --- 2. Fetch All Hotels ---
        all_hotels = hotels.get_hotels_near(
            lat=float(lat), 
            lon=float(lon), 
            check_in=check_in, 
            check_out=check_out,
            limit=50 
        )
        print(all_hotels)
        if not all_hotels:
            return jsonify([]), 200 # Return empty list if no hotels found

        # --- 3. Determine Price Limit and Filter Hotels ---
        limit = get_price_limit_hotels(all_hotels, bud_type)
        
        if limit <= 0:
             return jsonify([]), 200 # Return empty list if limit is invalid

        
        final_hotels_raw = []
        for h in all_hotels:
            try:
                current_price = float(h.get('price')) 
                h['price'] = current_price # Ensure numeric price
                
                if current_price <= limit:
                    final_hotels_raw.append(h)
            except (ValueError, TypeError):
                continue
        
        # Sort the final list in DESCENDING order of price
        final_hotels_raw.sort(key=lambda x: float(x.get('price', 0)), reverse=True)

        # --- 4. Final Data Shaping (Filtering and Renaming Keys) ---
        # Define the keys from hotel_finder and their desired final names
        REQUIRED_KEYS_MAP = {
            "hotel_name": "name",
            "hotel_id": "id", 
            "latitude": "lat", 
            "longitude": "long", 
            "address": "address",
            "price": "price",
            "currency": "currency" # Keeping currency is best practice for price clarity
        }
        
        final_hotels_filtered = []
        for h_raw in final_hotels_raw:
            filtered_item = {}
            for raw_key, final_key in REQUIRED_KEYS_MAP.items():
                filtered_item[final_key] = h_raw.get(raw_key)
                
            final_hotels_filtered.append(filtered_item)
            

        # --- 5. Return Final List as JSON ---
        
        # This returns the list of hotel dictionaries directly, as requested.
        return jsonify({"hotels":final_hotels_filtered,"tasks":t_r}), 200

    except requests.exceptions.RequestException as e:
        print(f"External API error: {e}")
        return jsonify({"error": "External data service unavailable."}), 503 

    except Exception as e:
        print(f"Internal error in hotel_option: {e}")
        return jsonify({"error": f"Internal error in hotel_option: {e} ","tasks_id":t,"tasks":t_r}), 500


@app.route("/hotel_choice",methods=["POST"])
@login_required
def hotel_choice():
    data = request.get_json()
    trip_id = data.get('trip_id')
    if not trip_id:
        return jsonify({"errors": {"trip": "trip_id is missing from request."}}), 400

    # 2. Find the trip in the database
    trip = db.session.get(Trip, trip_id)
    trip.hotel_name=data['hotel_name']
    trip.hotel_address=data['hotel_address']
    trip.hotel_lat=data['hotel_lat']
    trip.hotel_long=data['hotel_long']
    trip.hotel_price=data['hotel_price']
    db.session.commit()

    return jsonify({"message": "Choice updated successfully."}), 200


from datetime import time,timedelta
import datetime
import ilp_sol
import math

# --- Helper: Time Parser for Sorting ---
def parse_start_time(time_str):
    """Converts '09:30 AM' -> 570 minutes for sorting"""
    try:
        # Expected format "HH:MM AM/PM"
        t = pd.to_datetime(time_str, format='%I:%M %p')
        return t.hour * 60 + t.minute
    except:
        return 9999 # Push to end if format fails


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# --- Helper: Update Cluster Stats ---
def update_cluster_stats(cluster_id, cluster_data, all_places_dict):
    """
    Recalculates avg_score and mandatory_count for a specific cluster 
    after a new place has been added.
    """
    place_ids = cluster_data[cluster_id]['place_ids']
    if not place_ids:
        return

    total_score = 0
    mandatory_count = 0
    
    for pid in place_ids:
        place = all_places_dict.get(pid)
        if place:
            # Handle potential different score keys
            score = place.get('smart_score', place.get('score', 0))
            total_score += score
            if place.get('is_mandatory'):
                mandatory_count += 1
    
    new_avg = total_score / len(place_ids)
    
    # Update the data structure directly
    cluster_data[cluster_id]['stats']['avg_score'] = round(new_avg, 2)
    cluster_data[cluster_id]['stats']['mandatory_count'] = mandatory_count
    # Note: We are not recalculating 'nearest_clusters' here to save time, 
    # as centroid shift is usually negligible for one place.


@app.route("/ilp_solver",methods=["POST"])
@login_required
def ilp_solver():
    print("started")
    data = request.get_json()
    trip_id = data.get('trip_id')
    trip = db.session.get(Trip, trip_id)
    # start=trip.start_date
    # end=trip.end_date
    t_price=0
    h_price=0
    daily_timings=[]
    thre_time = time(14, 30, 0)
    if trip.budget_type is not None:
        task_result = AsyncResult(t["get_travel_cost"], app=tasks.celery)
        daily_price = task_result.result["final_cost"]
    else:
        if trip.needs_transport:
            if trip.up_mode.lower()=="flight":
                u=flight[(flight['flight_id'] == trip.up_mode_id) & 
                        (flight['source_city'] == trip.origin_city) & 
                        (flight['destination_city'] == trip.destination_city)]
            elif trip.up_mode.lower()=="train":
                u=train[(train['train_id'] == trip.up_mode_id) & 
                        (train['source_city'] == trip.origin_city) & 
                        (train['destination_city'] == trip.destination_city)]
            else:
                u=bus[(bus['bus_id'] == trip.up_mode_id) & 
                        (bus['source_city'] == trip.origin_city) & 
                        (bus['destination_city'] == trip.destination_city)]   
            if trip.down_mode.lower()=="flight":
                d=flight[(flight['flight_id'] == trip.down_mode_id) & 
                        (flight['source_city'] == trip.destination_city) & 
                        (flight['destination_city'] == trip.origin_city)]
            elif trip.down_mode.lower()=="train":
                d=train[(train['train_id'] == trip.down_mode_id) & 
                        (train['source_city'] == trip.destination_city) & 
                        (train['destination_city'] == trip.origin_city)]
            else:
                d=bus[(bus['bus_id'] == trip.down_mode_id) & 
                        (bus['source_city'] == trip.destination_city) & 
                        (bus['destination_city'] == trip.origin_city)]  
            t_price=u["price"].iloc[0]+d["price"].iloc[0]
            # print(pd.to_datetime(u["arrival_time"].iloc[0], format='%H:%M:%S').time())
            # print(pd.to_datetime(u["arrival_time"].iloc[0], format='%H:%M:%S').time())
            # if pd.to_datetime(u["arrival_time"].iloc[0], format='%H:%M:%S').time()>thre_time:
            #     print(u["arrival_time"].iloc[0])
            #     start = start + timedelta(days=1)
            # if pd.to_datetime(u["arrival_time"].iloc[0], format='%H:%M:%S').time()<thre_time:
            #     print(end)
            #     end=end-timedelta(days=1)
            #     print(end)

        if trip.needs_accommodation:
            h_price=trip.hotel_price
        d=trip.end_date-trip.start_date
        daily_price=((trip.budget_amount//trip.num_people)-t_price)//(d.days+1)-h_price
    
    if trip.needs_transport:
        if trip.up_mode.lower()=="flight":
            u=flight[(flight['flight_id'] == trip.up_mode_id) & 
                    (flight['source_city'] == trip.origin_city) & 
                    (flight['destination_city'] == trip.destination_city)&
                    (flight['departure_date']==trip.start_date.strftime('%Y-%m-%d'))]
        elif trip.up_mode.lower()=="train":
            u=train[(train['train_id'] == trip.up_mode_id) & 
                    (train['source_city'] == trip.origin_city) & 
                    (train['destination_city'] == trip.destination_city)&
                    (train['departure_date']==trip.start_date.strftime('%Y-%m-%d'))]
        else:
            u=bus[(bus['bus_id'] == trip.up_mode_id) & 
                    (bus['source_city'] == trip.origin_city) & 
                    (bus['destination_city'] == trip.destination_city)&
                    (bus['departure_date']==trip.start_date.strftime('%Y-%m-%d'))]   
        if trip.down_mode.lower()=="flight":
            d=flight[(flight['flight_id'] == trip.down_mode_id) & 
                    (flight['source_city'] == trip.destination_city) & 
                    (flight['destination_city'] == trip.origin_city)&
                    (flight['departure_date']==trip.end_date.strftime('%Y-%m-%d'))]
        elif trip.down_mode.lower()=="train":
            d=train[(train['train_id'] == trip.down_mode_id) & 
                    (train['source_city'] == trip.destination_city) & 
                    (train['destination_city'] == trip.origin_city)&
                    (train['departure_date']==trip.end_date.strftime('%Y-%m-%d'))]
        else:
            d=bus[(bus['bus_id'] == trip.down_mode_id) & 
                    (bus['source_city'] == trip.destination_city) & 
                    (bus['destination_city'] == trip.origin_city)&
                    (bus['departure_date']==trip.end_date.strftime('%Y-%m-%d'))]  
        start = pd.to_datetime(u["arrival_date"].iloc[0]).date()
        end = pd.to_datetime(d["departure_date"].iloc[0]).date()
        arrival_dt = pd.to_datetime(u["arrival_time"].iloc[0], format='%H:%M:%S') + timedelta(hours=2)
        if pd.to_datetime(u["arrival_time"].iloc[0], format='%H:%M:%S').time()>thre_time:
            start = start + timedelta(days=1)
            current_processing_date = start
        else:
            daily_timings.append([arrival_dt.hour * 60 + arrival_dt.minute, 20.5 * 60])
            current_processing_date = start + timedelta(days=1)

        # Loop until we reach the day BEFORE departure
        while current_processing_date < end:
            # Append Full Day Timings
            # 9.5 * 60  = 9:30 AM (570 mins)
            # 20.5 * 60 = 8:30 PM (1230 mins) -> CORRECTED from 8.5 (8:30 AM)
            daily_timings.append([9.5 * 60, 20.5 * 60])
            
            # Move to next day
            current_processing_date += timedelta(days=1)
        dep_dt = pd.to_datetime(d["departure_time"].iloc[0], format='%H:%M:%S') - timedelta(hours=2)
        if pd.to_datetime(d["departure_time"].iloc[0], format='%H:%M:%S').time()<thre_time:
            end = end - timedelta(days=1)
        else:
            daily_timings.append([9*60,dep_dt.hour * 60 + dep_dt.minute])
    

    else:
        start=trip.start_date
        end=trip.end_date
        current_processing_date = start 

        # Loop until we reach the day BEFORE departure
        while current_processing_date <=end:
            # Append Full Day Timings
            # 9.5 * 60  = 9:30 AM (570 mins)
            # 20.5 * 60 = 8:30 PM (1230 mins) -> CORRECTED from 8.5 (8:30 AM)
            daily_timings.append([9.5 * 60, 20.5 * 60])
            
            # Move to next day
            current_processing_date += timedelta(days=1)


    current_date = start
    final_itinerary = {}
    
    # 1. Create a lookup for places by ID for easy access
    all_places_dict = {p['id']: p for p in t_r['Places']}
    
    # 2. Identify available clusters (keys are strings "0", "1", etc.)
    # We maintain a list of cluster IDs that haven't been processed yet
    available_cluster_ids = list(t_r['clustering'].keys())
    print("each day loop started")
    # 3. Iterate through days
    day_index=0
    print(current_date)
    print(end)
    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        day_name = current_date.strftime("%A")
        print(f"\n--- Processing: {date_str} | {day_name} ---")
        
        if not available_cluster_ids:
            print("No more clusters available for this day.")
            final_itinerary[date_str] = []
            current_date += timedelta(days=1)
            continue

        # --- STEP A: Select Best Cluster ---
        # Logic: Pick the available cluster with the MAX avg_score
        best_cluster_id = max(
            available_cluster_ids, 
            key=lambda cid: t_r['clustering'][cid]['stats']['avg_score']
        )
        
        print(f"Selected Cluster ID: {best_cluster_id} (Score: {t_r['clustering'][best_cluster_id]['stats']['avg_score']})")
        
        # Get places for this cluster
        current_cluster_place_ids = t_r['clustering'][best_cluster_id]['place_ids']
        day_places_data = [all_places_dict[pid] for pid in current_cluster_place_ids]
        print(day_places_data)
        # --- STEP B: Run ILP Solver ---
        # Call your external function here
        print("first ilp")
        final_schedule, unvisited_ids = ilp_sol.generate_itinerary(day_places_data,trip.hotel_name,trip.hotel_lat,trip.hotel_long,day_name,daily_price,
                                                                     trip.pace,daily_timings[day_index][0],daily_timings[day_index][1],trip.food_preference)
        print("check done")
        if isinstance(final_schedule["timeline"], dict):
            print("Converting dictionary schedule to list...")
            temp_list = []
            for time_key, place_data in final_schedule["timeline"].items():
                # If 'start_time' is missing, grab it from the key (e.g., "09:00 AM-10:00 AM")
                if 'start_time' not in place_data:
                    place_data['start_time'] = time_key.split('-')[0].strip()
                temp_list.append(place_data)
            final_schedule["timeline"] = temp_list
        final_schedule["timeline"].sort(key=lambda x: parse_start_time(x.get('start_time', '11:59 PM')))
        print(day_name+"  ilp success")
        # Store successful itinerary
        final_itinerary[date_str] = {
            "day": day_name,
            "date":date_str,
            "places": final_schedule["timeline"],
            "meals":final_schedule["restaurants"]
        }

        # --- STEP C: Handle Unvisited Mandatory Places ---
        if unvisited_ids:
            print(f"Unvisited IDs: {unvisited_ids}")
            
            # Remove current cluster from available list so we don't reassign back to it
            # (We do this temporarily to find the *next* best cluster)
            remaining_clusters = [cid for cid in available_cluster_ids if cid != best_cluster_id]
            
            if remaining_clusters:
                for uid in unvisited_ids:
                    place_obj = all_places_dict[uid]
                    
                    # Only care if it's mandatory
                    if place_obj.get('is_mandatory'):
                        print(f"  > Reassigning Mandatory Place: {place_obj['name']}")
                        
                        # Find nearest cluster
                        p_lat = place_obj['latitude']
                        p_long = place_obj['longitude']
                        
                        best_target_id = None
                        min_dist = float('inf')
                        
                        for target_id in remaining_clusters:
                            # Get centroid of target cluster
                            # Ensure your centroid structure matches: clustered_data['centroid']['values'][target_id]
                            c_vals = t_r['centroid']['values'].get(str(target_id))
                            if not c_vals: continue
                                
                            dist = calculate_distance(p_lat, p_long, c_vals['lat'], c_vals['long'])
                            
                            if dist < min_dist:
                                min_dist = dist
                                best_target_id = target_id
                        
                        if best_target_id:
                            print(f"    -> Moved to Cluster {best_target_id} (Dist: {min_dist:.2f}km)")
                            # 1. Add to new cluster's list
                            t_r['clustering'][best_target_id]['place_ids'].append(uid)
                            # 2. Update stats for that cluster (so selection logic sees new score)
                            update_cluster_stats(best_target_id, t_r['clustering'], all_places_dict)
                            # 3. Update the place's own cluster label for consistency
                            place_obj['cluster_label'] = int(best_target_id)
            else:
                print("  > Warning: No remaining days to reassign mandatory places.")

        # --- STEP D: Finalize Day ---
        # Permanently remove the processed cluster from availability
        available_cluster_ids.remove(best_cluster_id)
        current_date += timedelta(days=1)
        print(day_name+"  success clsuter reordering")
        existing_itinerary = Itinerary.query.filter_by(trip_id=trip_id).first()

        if existing_itinerary:
            print(f"🔄 Itinerary exists for Trip {trip_id}. Updating...")
            # UPDATE the existing row
            existing_itinerary.daily_plan = final_itinerary# Update timestamp
        else:
            print(f"💾 Saving NEW itinerary for Trip {trip_id}...")
            # CREATE a new row
            new_itinerary = Itinerary(
                trip_id=trip_id,
                daily_plan=final_itinerary
            )
        db.session.add(new_itinerary)
        db.session.commit()
    return jsonify(final_itinerary), 200




@app.route("/logout", methods=["POST"])
@login_required  # Ensures only logged-in users can access this
def logout():
    # *** THIS IS THE KEY FLASK-LOGIN FUNCTION ***
    # It clears the session and cookie
    logout_user()
    return jsonify({"message": "Logged out successfully."}), 200


# --- RUN THE APP ---
if __name__ == "__main__":
    # Create the database and tables before first request
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)