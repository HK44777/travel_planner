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

import pandas as pd
import numpy as np
# CHANGED: Removed FlaskForm, wtforms, and CSRFProtect imports

bus=pd.read_excel('Bus_Timings_dataset.xlsx')
flight=pd.read_excel('flight_data.xlsx')
train=pd.read_excel('Train_Timings_dataset.xlsx')


app = Flask(__name__)
# You MUST change this secret key in production!
# It's required for Flask-Login sessions
app.config["SECRET_KEY"] = "123546879"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
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
    mode=db.Column(db.String(100), nullable=True)
    mode_id=db.Column(db.String(100), nullable=True)
    hotel_id=db.Column(db.String(100), nullable=True)

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

        # 3. Add to the database
        db.session.add(new_trip)
        db.session.commit()

        # 4. Send a success response
        return jsonify({"message": "Trip created successfully!", "trip_id": new_trip.id}), 201

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
    


@app.route("/transport_option",methods=["POST"])
@login_required
def transport_option():
    data=request.get_json()
    trip_id=data.get('trip_id')
    trip = db.session.get(Trip, trip_id)
    a_d=data.get('a_d')
    if a_d=='a':
        src=trip.origin_city
        dest=trip.destination_city
        start_date=trip.start_date
    else:
        src=trip.destination_city
        dest=trip.origin_city
        start_date=trip.end_date
    bud_type=trip.transport_type
    ppl=trip.num_people

    
    bus['departure_date'] = pd.to_datetime(bus['departure_date'])
    start_date = pd.to_datetime(start_date).date()

    # Filter the DataFrame
    filtered_buses = bus[
        (bus['source_city'] == src) &
        (bus['destination_city'] == dest) &
        (bus['departure_date'].dt.date == start_date) &
        (bus['availability_seats'] >= ppl)
    ]
    
    # Format the output as a list of lists of tuples
    bus_result = []
    for index, row in filtered_buses.iterrows():
        bus_result.append([(row['bus_id'], row['price_INR'])])

    budget_mapping = {
        'basic': 1,
        'economy': 2,
        'standard': 3,
        'premium': 4,
        'luxury': 5
    }

    # Get the multiplier for the given budget type
    multiplier = budget_mapping.get(bud_type.lower())


    # Extract all prices to find min and max
    prices = [item[0][1] for item in bus_result] # bus_data is list of lists of tuples, e.g. [[('VB-5001', 1196)]]
    min_price = min(prices)
    max_price = max(prices)
    price_difference = max_price - min_price

    # Calculate the upper price limit based on the budget type formula
    # (multiplier / 5) * difference + min_price
    upper_price_limit = (multiplier / 5) * price_difference + min_price

    filtered_b = []
    for bus_tuple_list in bus_result:
        bus_id, price = bus_tuple_list[0] # Assuming each inner list has one tuple
        if price <= upper_price_limit:
            filtered_b.append(bus_id)
    final_buses=filtered_buses[(bus['bus_id'].isin(filtered_b))]
    
    train.columns = train.columns.str.strip()
    train['departure_date'] = pd.to_datetime(train['departure_date'])

    # Filter the DataFrame
    filtered_trains = train[
        (train['source_city'] == src) &
        (train['destination_city'] == dest) &
        (train['departure_date'].dt.date == start_date) &
        (train['availability_seats'] >= ppl)
    ]
    
    # Format the output as a list of lists of tuples
    train_result = []
    for index, row in filtered_trains.iterrows():
        train_result.append([(row['train_id'], row['price_INR'])])

    # Extract all prices to find min and max
    prices = [item[0][1] for item in train_result] # bus_data is list of lists of tuples, e.g. [[('VB-5001', 1196)]]
    min_price = min(prices)
    max_price = max(prices)
    price_difference = max_price - min_price

    # Calculate the upper price limit based on the budget type formula
    # (multiplier / 5) * difference + min_price
    upper_price_limit = (multiplier / 5) * price_difference + min_price

    filtered_t = []
    for train_tuple_list in train_result:
        train_id, price = train_tuple_list[0] # Assuming each inner list has one tuple
        if price <= upper_price_limit:
            filtered_t.append(train_id)
    final_trains=filtered_trains[(train['train_id'].isin(filtered_t))]

    flight['departure_date'] = pd.to_datetime(flight['departure_date'])

    # Filter the DataFrame
    filtered_flights = flight[
        (flight['source_city'] == src) &
        (flight['destination_city'] == dest) &
        (flight['departure_date'].dt.date == start_date) &
        (flight['availability_seats'] >= ppl)
    ]
    
    # Format the output as a list of lists of tuples
    flight_result = []
    for index, row in filtered_flights.iterrows():
        flight_result.append([(row['flight_id'], row['price_INR'])])

    # Extract all prices to find min and max
    prices = [item[0][1] for item in flight_result] # bus_data is list of lists of tuples, e.g. [[('VB-5001', 1196)]]
    min_price = min(prices)
    max_price = max(prices)
    price_difference = max_price - min_price

    # Calculate the upper price limit based on the budget type formula
    # (multiplier / 5) * difference + min_price
    upper_price_limit = (multiplier / 5) * price_difference + min_price

    filtered_f = []
    for flight_tuple_list in flight_result:
        flight_id, price = flight_tuple_list[0] # Assuming each inner list has one tuple
        if price <= upper_price_limit:
            filtered_f.append(flight_id)
    final_flights=filtered_flights[(flight['flight_id'].isin(filtered_f))]

    bus_data = final_buses.to_dict(orient='records') 
    train_data = final_trains.to_dict(orient='records') 
    flight_data = final_flights.to_dict(orient='records') 

    # 2. Create the Master Dictionary
    master_data = {
        "bus": bus_data,
        "train": train_data,
        "flight": flight_data
    }
    json_output = json.dumps(master_data, indent=4, default=str)
    return jsonify(json_output), 201
    



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
        db.session.commit()
        
        return jsonify({"message": "Trip details updated successfully!", "trip_id": trip.id}), 200

    except ValidationError as e:
        # If validation fails, send back the formatted errors
        return jsonify({"errors": format_pydantic_errors(e)}), 400

    except Exception as e:
        # Catch any other potential errors (like database errors)
        db.session.rollback()
        return jsonify({"errors": {"database": str(e)}}), 500


@app.route("/transport_choice",methods=["POST"])
@login_required
def transport_choice():
    data = request.get_json()
    trip_id = data.get('trip_id')
    if not trip_id:
        return jsonify({"errors": {"trip": "trip_id is missing from request."}}), 400

    # 2. Find the trip in the database
    trip = db.session.get(Trip, trip_id)
    trip.mode=data['mode']
    trip.mode_id=data['mode_id']
    return jsonify({"message": "Choice updated successfully."}), 200


@app.route("/hotel_choice",methods=["POST"])
@login_required
def hotel_choice():
    data = request.get_json()
    trip_id = data.get('trip_id')
    if not trip_id:
        return jsonify({"errors": {"trip": "trip_id is missing from request."}}), 400

    # 2. Find the trip in the database
    trip = db.session.get(Trip, trip_id)
    trip.hotel_id=data['hotel_id']
    return jsonify({"message": "Choice updated successfully."}), 200


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