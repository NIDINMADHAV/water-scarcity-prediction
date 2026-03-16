from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import pandas as pd
import json
import plotly.express as px
import plotly

# ---------------- APP CONFIG ----------------
app = Flask(__name__)
app.secret_key = "your_secret_key"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- LOGIN MANAGER ----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.unauthorized_handler
def unauthorized():
    flash("Please login to access this page", "warning")
    return redirect(url_for('login'))

# ---------------- USER MODEL ----------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    country = db.Column(db.String(100))
    year = db.Column(db.Integer)
    result = db.Column(db.String(100))

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ---------------- LOAD ML FILES ----------------
model = pickle.load(open("model.pkl", "rb"))
le_country = pickle.load(open("le_Country.pkl", "rb"))
le_scarcity = pickle.load(open("le_Water_Scarcity_Level.pkl", "rb"))

df = pd.read_csv("cleaned_global_water_consumption.csv")

# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template('home.html')

# ---------- REGISTER ----------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            flash("Email already exists", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        user = User(
            username=username,
            email=email,
            password=hashed_password
        )

        db.session.add(user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

# ---------- LOGIN ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password", "danger")

    return render_template('login.html')

# ---------- LOGOUT ----------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully", "info")
    return redirect(url_for('home'))

# ---------- PREDICT ----------
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        Country = request.form['Country']

        try:
            Year = int(request.form['Year'])
            Total_Water = float(request.form['Total_Water'])
            Per_Capita = float(request.form['Per_Capita'])
            Agri_Water = float(request.form['Agri_Water'])
            Industrial_Water = float(request.form['Industrial_Water'])
            Household_Water = float(request.form['Household_Water'])
            Rainfall = float(request.form['Rainfall'])
            Groundwater = float(request.form['Groundwater'])
        except ValueError:
            flash("Please enter valid numeric values", "danger")
            return redirect(url_for('predict'))
    
        # enforce valid ranges
        if Year < 1900 or Year > 2100:
            flash("Year must be between 1900 and 2100", "danger")
            return redirect(url_for('predict'))

        try:
            Country_encoded = le_country.transform([Country])[0]
        except:
            Country_encoded = 0

        features = np.array([[
            Country_encoded, Year, Total_Water, Per_Capita,
            Agri_Water, Industrial_Water, Household_Water,
            Rainfall, Groundwater
        ]])

        pred_encoded = model.predict(features)[0]
        prediction = le_scarcity.inverse_transform([pred_encoded])[0]
        new_pred = Prediction(
            user_id=current_user.id,
            country=Country,
            year=Year,
            result=prediction
        )

        db.session.add(new_pred)
        db.session.commit()
        
        prob = model.predict_proba(features)
        confidence = round(np.max(prob) * 100, 2)
        
        return render_template(
            'result.html',
            prediction=prediction,
            confidence=confidence
        )

    countries = le_country.classes_
    return render_template('predict.html', countries=countries)

# ---------- DASHBOARD ----------
@app.route('/dashboard')
@login_required
def dashboard():
    models = [
        'Naive Bayes', 'Logistic Regression', 'SVM',
        'Random Forest', 'XGBoost', 'Decision Tree',
        'AdaBoost', 'Gradient Boost', 'KNN'
    ]

    accuracies = [0.78, 0.82, 0.85, 0.88, 0.90, 0.83, 0.84, 0.86, 0.81]

    # Bar Chart
    bar_data = df.groupby('Country')['Water Scarcity Level'] \
                 .value_counts().reset_index(name='Count')

    fig_bar = px.bar(
        bar_data,
        x='Country',
        y='Count',
        color='Water Scarcity Level',
        title='Water Scarcity by Country'
    )

    barJSON = json.dumps(fig_bar, cls=plotly.utils.PlotlyJSONEncoder)

    # Choropleth Map
    latest_df = df.sort_values('Year').groupby('Country').tail(1)

    fig_map = px.choropleth(
        latest_df,
        locations='Country',
        locationmode='country names',
        color='Water Scarcity Level',
        color_continuous_scale='Reds',
        title='World Water Scarcity Map'
    )

    mapJSON = json.dumps(fig_map, cls=plotly.utils.PlotlyJSONEncoder)

    history = Prediction.query.filter_by(user_id=current_user.id).all()

    total_predictions = len(df)

    most_country = df['Country'].value_counts().idxmax()
    
    avg_accuracy = round(sum(accuracies) / len(accuracies), 2)

    return render_template(
        'dashboard.html',
        models=models,
        accuracies=accuracies,
        barJSON=barJSON,
        mapJSON=mapJSON,
        history=history,
        total_predictions=total_predictions,
        most_country=most_country,
        avg_accuracy=avg_accuracy
    )

# ---------- ABOUT ----------
@app.route('/about')
def about():
    return render_template('about.html')

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)
