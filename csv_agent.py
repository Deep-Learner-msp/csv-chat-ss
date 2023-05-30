import pandas as pd
from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from flask import Flask, request, jsonify
# from dotenv import load_dotenv
import re
import os
# load_dotenv()

app = Flask(__name__)ßß

df = pd.read_csv('six_month_yahoo_finance.csv')

csv = 'six_month_yahoo_finance.csv'

agent = create_csv_agent(ChatOpenAI(temperature=0,engine="ss-gpt-32k"), csv)

identity_patterns = [
    "who\s*(are|r)\s*you\??",
    "who\s*(did|has)?\s*(designed|designd|desgined)\s*you\??",
    "who\s*(dev|devloped|developed)\s*(you|u)\??",
    "what('s|s| is)?\s*your\s*(name|duty)\??",
    "what\s*do\s*you\s*do\??",
    "tell\s*me\s*about\s*your\s*self\??",
    "describe\s*your\s*self\??",
]


@app.route("/query", methods=["POST"])
def query():
    try:
        message = request.json.get("message")
        if not message:
            raise ValueError("No message provided in the request.")
        
        # Check for specific queries
        if any(re.fullmatch(pattern, message, re.IGNORECASE) for pattern in identity_patterns):
            response = "Greetings! I'm SheetGPT, a product of State Street Bionics team. My architecture is empowered by OpenAI's advanced Generative AI technology. My primary responsibility involves analyzing tabular data and providing valuable insights for users. Rest assured, your data-driven queries are in capable hands."
        else:
            # Generate a response from the LLM
            response = agent.run(message)
        
        # Return the LLM's response
        return jsonify({"response": response})
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": "SheetGPT is busy in serving other requests at this moment, please try after sometime."}), 500

@app.route("/data", methods=["GET"])
def data():
    try:
        data = df.dropna()
        # Convert the dataframe to a list of dictionaries for JSON serialization
        data = df.to_dict(orient="records")
        
        # Return the data
        return jsonify({"data": data})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
