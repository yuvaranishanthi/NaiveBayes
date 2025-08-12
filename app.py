from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""
    if request.method == "POST":
        message = request.form["message"]
        data = vectorizer.transform([message])
        pred = model.predict(data)[0]
        prediction = "🚨 Spam" if pred == 1 else "✅ Ham (Not Spam)"
    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)
