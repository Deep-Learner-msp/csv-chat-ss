import pandas as pd
from pandasai import PandasAI
from pandasai.llm.azure_openai import AzureOpenAI
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import re
import openai
load_dotenv()

app = Flask(__name__)
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview" 

df = pd.read_csv('six_month_yahoo_finance.csv')

llm = AzureOpenAI(api_version="2023-03-15-preview", deployment_name="ss-gpt")
pandas_ai = PandasAI(llm, verbose=True)

identity_patterns = [
    "who\s*(are|r)\s*you\??",
    "who\s*(did|has)?\s*designed\s*you\??",
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
            response = pandas_ai.run(df, prompt=message)
        
        # Return the LLM's response
        return jsonify({"response": response})
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": "SheetGPT is busy in serving other requests at thi moment, please after sometime, bye."}), 500

@app.route("/data", methods=["GET"])
def data():
    try:
        # Convert the dataframe to a list of dictionaries for JSON serialization
        data = df.to_dict(orient="records")
        
        # Return the data
        return jsonify({"data": data})
    except Exception as e:
        app.logger.exception(e)
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
