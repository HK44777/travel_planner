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
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired, Email, Length
from flask_cors import CORS 
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
# You MUST change this secret key in production!
# It's required for Flask-Login sessions
app.config["SECRET_KEY"] = "123546879"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["WTF_CSRF_TIME_LIMIT"] = 3600 
csrf = CSRFProtect(app)

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


# --- VALIDATION FORMS (using Flask-WTF) ---
# We use this to validate the JSON data from React

class SignUpForm(FlaskForm):
    class Meta:
        csrf = False
    name = StringField("name", validators=[InputRequired()])
    email = StringField(
        "email", validators=[InputRequired(), Email(message="Invalid email address.")]
    )
    password = PasswordField(
        "password",
        validators=[
            InputRequired(),
            Length(min=8, message="Password must be at least 8 characters long."),
        ],
    )

class LoginForm(FlaskForm):
    class Meta:
        csrf = False
    email = StringField("email", validators=[InputRequired(), Email()])
    password = PasswordField("password", validators=[InputRequired()])

# --- API ENDPOINTS ---

# A simple "hello world" route to check if the server is up
@app.route("/")
def hello():
    return jsonify({"message": "Flask server is running!"})

@app.route("/signup", methods=["POST"])
@csrf.exempt
def signup():
    # Get JSON data from the React request
    data = request.get_json()
    # Pass the data to the form for validation
    form = SignUpForm(data=data)

    if form.validate():
        # Check if email already exists
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            return jsonify({"errors": {"email": ["Email address already in use."]}}), 400

        # Create new user
        new_user = User(name=form.name.data, email=form.email.data)
        new_user.set_password(form.password.data)  # This hashes the password

        # Add to database
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": f"User {new_user.name} created successfully!"}), 201
    else:
        # If validation fails, return the specific errors
        # This is what your React app will read
        return jsonify({"errors": form.errors}), 400


@app.route("/login", methods=["POST"])
@csrf.exempt
def login():
    data = request.get_json()
    form = LoginForm(data=data)

    if form.validate():
        email = form.email.data
        password = form.password.data

        # Find the user by email
        user = User.query.filter_by(email=email).first()

        # Check if user exists and if the password is correct
        if user and user.check_password(password):
            # *** THIS IS THE KEY FLASK-LOGIN FUNCTION ***
            # It creates the session and sends the 'Set-Cookie' header
            login_user(user) 
            
            return jsonify({
                "message": f"Welcome back, {user.name}!",
                "user": {"id": user.id, "name": user.name, "email": user.email}
            }), 200
        else:
            # Failed login
            return jsonify({"errors": {"auth": ["Invalid email or password."]}}), 401
    else:
        # Failed form validation (e.g., missing email)
        return jsonify({"errors": form.errors}), 400


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