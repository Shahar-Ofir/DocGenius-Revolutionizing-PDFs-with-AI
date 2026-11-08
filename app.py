import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as gexc
from google.api_core import retry as gretry

# Load the .env file
load_dotenv()

# Retrieve the API key from the environment variable
GEMINI_API = os.getenv("GEMINI_API")
genai.configure(api_key=GEMINI_API)

# Production limits for model calls (generation is read-only, so retries are safe)
MODEL_TIMEOUT_S = 60
MAX_OUTPUT_TOKENS = 1024
_RETRY = gretry.Retry(
    predicate=gretry.if_exception_type(
        gexc.ResourceExhausted, gexc.ServiceUnavailable,
        gexc.DeadlineExceeded, gexc.InternalServerError,
    ),
    initial=1.0, maximum=10.0, multiplier=2.0, timeout=MODEL_TIMEOUT_S * 2,
)

# Streamlit app
st.title("Gemini AI Content Generator")

# Text input for user prompt (with form submission on Enter)
user_input = st.text_input("Enter a prompt:")

# Check if user input is provided and the 'Enter' key is pressed
if user_input:
    # Use the Gemini model to generate content
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(
            user_input,
            generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS},
            request_options={"timeout": MODEL_TIMEOUT_S, "retry": _RETRY},
        )
        # Display the generated response
        st.write("Response from Gemini AI:")
        st.write(response.text)
    except Exception as e:
        st.error(f"Gemini request failed: {e}")
else:
    st.write("Please enter a prompt to generate content.")