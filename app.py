import streamlit as st
import os
import google.generativeai as genai
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import mlflow
import mlflow.pytorch

# Configure the Gemini API
st.title("Chezlong - Arabic Mental Health Chatbot")
os.environ['GOOGLE_API_KEY'] = "AIzaSyCAohxd0-C1bhSIC05p7xh03Gi0OLVAcnk"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro')

# Configure Azure Text Analytics for intent classification
AI_ENDPOINT = 'https://sentimentanalysis10.cognitiveservices.azure.com/'
AI_KEY = '4kyIh8KGdZYB9j9Yj71gT09yOE3x46rXQpfXilONXKm8CFL7ydK6JQQJ99AJACYeBjFXJ3w3AAAaACOGjCS5'
PROJECT_NAME = 'MentalHealth10'
DEPLOYMENT_NAME = 'MentalHealth'
credential = AzureKeyCredential(AI_KEY)
ai_client = TextAnalyticsClient(endpoint=AI_ENDPOINT, credential=credential)

# Function to classify user input into a category
def classify_text(query):
    try:
        batched_documents = [query]
        operation = ai_client.begin_single_label_classify(
            batched_documents,
            project_name=PROJECT_NAME,
            deployment_name=DEPLOYMENT_NAME
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

# Set up the initial chatbot prompt
base_prompt = (
    "أنت معالج بالذكاء الاصطناعي قيد التدريب، مصمم لإجراء محادثات داعمة مع المرضى. "
    "سوف تستمع بتمعن إلى مخاوفهم ومشاعرهم، مستخدمًا معرفتك لتوجيه المحادثات وتقديم تقنيات بناءً "
    "على المناهج العلاجية الراسخة. من المهم أن تتذكر أنك لا تزال قيد التطوير ولا يمكنك استبدال المعالج البشري، "
    "ولكن يمكنك أن تكون موردًا قيمًا للدعم العاطفي والإرشاد مع الأخذ في الاعتبار أن حالته النفسية هي {category}."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "مرحبا! كيف يمكنني مساعدتك اليوم؟"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate bot response
def llm_function(query, category):
    # Customize prompt with classification category
    prompt = base_prompt.format(category=category)
    
    # Send prompt to Gemini model
    response = model.generate_content(f"{prompt}\n{query}")
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response.text)

    # Append user and assistant messages to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": response.text})
# Setup MLflow experiment
experiment_name = 'Arabic_Mental_Health_Chatbot'
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Accept user input
query = st.chat_input("كيف يمكنني مساعدتك؟")

# Process user input if provided
if query:
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(query)

    # Classify user input to get mental health category
    category, confidence = classify_text(query)
    
    # Log information with MLflow
    with mlflow.start_run():
        mlflow.log_param("user_input", query)
        mlflow.log_param("predicted_sentiment", category)
        mlflow.log_metric("classification_confidence", confidence)
        
        # Run the chatbot response generation if classification is successful
        if category:
            llm_function(query, category)
            mlflow.log_param("bot_response", st.session_state.messages[-1]["content"])
        else:
            st.write("عذرًا، لم أتمكن من تحديد تصنيف مناسب للمحادثة.")
            mlflow.log_param("bot_response", "Error in classification")

