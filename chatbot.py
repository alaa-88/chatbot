import streamlit as st
import os
import google.generativeai as genai
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv

load_dotenv()
# Configure the Gemini API
st.title("Chezlong - Arabic Mental Health Chatbot")
genai_api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=genai_api_key)
model = genai.GenerativeModel('gemini-pro')

# Configure Azure Text Analytics for intent classification
ai_endpoint = os.getenv('AI_ENDPOINT')
ai_key = os.getenv('AI_KEY')
project_name = os.getenv('PROJECT_NAME')
deployment_name = os.getenv('DEPLOYMENT_NAME')
credential = AzureKeyCredential(ai_key)
ai_client = TextAnalyticsClient(endpoint=ai_endpoint, credential=credential)

 # Function to classify user input into a category
def classify_text(query):
    try:
        batched_documents = [query]
        operation = ai_client.begin_single_label_classify(
            batched_documents,
            project_name=project_name,
            deployment_name=deployment_name
        )
        document_results = operation.result()

        # Extract classification result
        for classification_result in document_results:
            if classification_result.kind == "CustomDocumentClassification":
                classification = classification_result.classifications[0]
                return classification.category, classification.confidence_score
            elif classification_result.is_error:
                return None, classification_result.error.message
    except Exception as ex:
        return None, str(ex)

# Define a base prompt for the bot
base_prompt = (
    "أنت معالج بالذكاء الاصطناعي قيد التدريب، مصمم لإجراء محادثات داعمة مع المرضى. "
    "سوف تستمع بتمعن إلى مخاوفهم ومشاعرهم، مستخدمًا معرفتك لتوجيه المحادثات وتقديم تقنيات "
    "بناءً على المناهج العلاجية الراسخة. من المهم أن تتذكر أنك لا تزال قيد التطوير ولا يمكنك استبدال "
    "المعالج البشري، ولكن يمكنك أن تكون موردًا قيمًا للدعم العاطفي والإرشاد مع الأخذ في الاعتبار أن "
    "حالته النفسية هي {category}."
)

# MLflow Custom Model Wrapper
class GeminiModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = genai.GenerativeModel('gemini-pro')  # Initialize Gemini model within MLflow environment

    def predict(self, context, model_input):
        category = model_input.get("category", "عام")  # Default category if not provided
        prompt = base_prompt.format(category=category)
        response = self.model.generate_content(f"{prompt}\n{model_input['text']}")
        return response.text

# Log model and parameters to MLflow
def log_gemini_model(user_message, bot_response, category, confidence):
    experiment_name = 'Arabic Mental Health Chatbot'
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters and model details
        mlflow.log_param("user_message", user_message)
        mlflow.log_param("predicted_sentiment", category)
        mlflow.log_param("confidence_score", confidence)
        mlflow.log_param("bot_response", bot_response)
        
        # Log the custom Gemini model as a pyfunc model
        mlflow.pyfunc.log_model(
            "gemini_model",
            python_model=GeminiModelWrapper(),
            conda_env={
                "channels": ["defaults"],
                "dependencies": [
                    "python=3.9",
                    "google-generativeai",  # Ensure this package is included in the conda environment
                    "mlflow",
                ],
                "name": "gemini_env"
            }
        )
    print("Gemini model and parameters logged successfully.")

# Chatbot session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحبا! كيف يمكنني مساعدتك اليوم؟"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat handling and bot response
def llm_function(query, category):
    prompt = base_prompt.format(category=category)
    response = model.generate_content(f"{prompt}\n{query}")

    # Add responses to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response.text})
    return response.text

# User input
query = st.chat_input("كيف يمكنني مساعدتك؟")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    # Classify the user's input and get the intent category and confidence score
    category, confidence = classify_text(query)
    if category:
        bot_response = llm_function(query, category)
    else:
        bot_response = "عذرًا، لم أتمكن من تحديد تصنيف مناسب للمحادثة."

    # Display response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # Log the interaction to MLflow
    log_gemini_model(query, bot_response, category, confidence)
