from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from math import ceil, floor

app = Flask(__name__)

# Baca dataset
data = pd.read_csv("uas_datascience.csv")

# Pilih fitur dan target
X = data.iloc[:, :13]
y = data.loc[:, "overall_rating"]

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model
rf_model = RandomForestRegressor()

# Latih model
rf_model.fit(X_train, y_train)

# Simpan model ke file
pickle.dump(rf_model, open("rf_model.pkl", "wb"))


# Fungsi untuk membulatkan nilai
@app.template_filter('custom_round')
def custom_round_filter(value):
    decimal_part = value - floor(value)
    if decimal_part >= 0.5:
        return ceil(value)
    else:
        return floor(value)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input from the form
        crossing = float(request.form['crossing'])
        finishing = float(request.form['finishing'])
        heading = float(request.form['heading'])
        short_passing = float(request.form['short_passing'])
        dribbling = float(request.form['dribbling'])
        freekick = float(request.form['freekick'])
        long_passing = float(request.form['long_passing'])
        ball_control = float(request.form['ball_control'])
        shot_power = float(request.form['shot_power'])
        stamina = float(request.form['stamina'])
        strength = float(request.form['strength'])
        interceptions = float(request.form['interceptions'])
        penalties = float(request.form['penalties'])

        # Get selected model
        selected_model = request.form['model']
        model_dict = {
            'Random Forest': rf_model,
        }

        selected_model = model_dict[selected_model]

        # Make a prediction using the selected model
        input_data = [[crossing, finishing, heading, short_passing, dribbling, freekick, long_passing, ball_control, shot_power, stamina, strength, interceptions, penalties]]
        prediction = selected_model.predict(input_data)

        # Render the result on the same HTML page
        return render_template('form.html', prediction=prediction[0])

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
