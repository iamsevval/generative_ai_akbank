import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(f"Loaded API Key ends with: ...{str(api_key)[-4:] if api_key else 'None'}")

if not api_key:
    print("API Key is missing!")
    exit(1)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key
)

try:
    response = llm.invoke("Hello, who are you?")
    print("Success! Response:", response.content)
except Exception as e:
    print("Error encountered:", type(e).__name__)
    print("Error details:", str(e))
