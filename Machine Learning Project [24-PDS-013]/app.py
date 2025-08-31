from flask import Flask, render_template, request, url_for
import joblib
import numpy as np

app = Flask(__name__)


model_path = r"C:\Users\Hp\OneDrive\Desktop\Machine Learning Project [24-PDS-013]\rf_model.pkl"
scaler_path = r"C:\Users\Hp\OneDrive\Desktop\Machine Learning Project [24-PDS-013]\scaler.pkl"
encoder_path = r"C:\Users\Hp\OneDrive\Desktop\Machine Learning Project [24-PDS-013]\label_encoders.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoder_path)


def encode_category(column_name, value):
    if value in label_encoders[column_name].classes_:
        return label_encoders[column_name].transform([value])[0]
    else:
        return label_encoders[column_name].transform(["Unknown"])[0]  

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        
        airline = request.form["airline"]
        traveller_type = request.form["traveller_type"]
        cabin = request.form["cabin"]
        seat_comfort = int(request.form["seat_comfort"])
        cabin_service = int(request.form["cabin_service"])
        food_bev = int(request.form["food_bev"])
        entertainment = int(request.form["entertainment"])
        value_for_money = int(request.form["value_for_money"])

        
        airline_encoded = encode_category("airline", airline)
        traveller_type_encoded = encode_category("traveller_type", traveller_type)
        cabin_encoded = encode_category("cabin", cabin)

        
        input_data = np.array([[airline_encoded, traveller_type_encoded, cabin_encoded,
                                seat_comfort, cabin_service, food_bev, entertainment, value_for_money]])
        
        
        input_scaled = scaler.transform(input_data)

        
        pred = model.predict(input_scaled)
        pred_prob = model.predict_proba(input_scaled)[0][1] * 100  

        
        prediction = "Yes, the passenger is likely to recommend!" if pred[0] == 1 else "No, the passenger is unlikely to recommend."
        probability = round(pred_prob, 2)

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
