import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)

llm = OpenAI(api_token="")
dataframe = pd.read_csv('six_month_yahoo_finance.csv')
pandas_ai = PandasAI(llm, verbose=True)

@app.route("/query", methods=["POST"])
def query():
    try:
        message = request.json.get("message")
        if not message:
            raise ValueError("No message provided in the request.")
        
        # Generate a response from the LLM
        response = pandas_ai.run(dataframe, prompt=message)

        # Return the LLM's response
        return jsonify({"response": response})
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
