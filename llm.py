from langchain_google_genai import ChatGoogleGenerativeAI
GOOGLE_API_KEY="AIzaSyCOpSLT4ie9pzRdSSTBwdq8o1o8UfPIm4M"
llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm.invoke("Sing a ballad of LangChain.")