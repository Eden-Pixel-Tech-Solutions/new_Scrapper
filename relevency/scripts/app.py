from flask import Flask, render_template, request, jsonify
from global_relevancy import predict   # your existing model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    result = predict(query, top_k=5)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
