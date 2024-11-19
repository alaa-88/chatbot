# **Project Title: Chatlong - Arabic Mental Health Chatbot**

## **Objective**:

The goal of this project is to develop an Arabic mental health chatbot named Chatlong. This chatbot aims to provide supportive conversations and guidance to users while acknowledging their mental state. It utilizes natural language processing (NLP) techniques to understand user inputs, classifies them using Azure Text Analytics, and generates contextually appropriate responses using a fine-tuned Meta-Llama-3-8B-Instruct model.


##### Check It Out!: [Chatlong - Arabic Mental Health Chatbot Streamlit App](https://chatlong-fclzmupbxfnpmvtrkng5qr.streamlit.app/)  

## **Usage Instructions**:

### 1. Fine-tuning and Data Preprocessing (Notebook)
* The Jupyter Notebook (`Arabic_Mental_Health_Chatbot_Fine-tuning.ipynb`) details the process of data preprocessing, model fine-tuning, and initial inference testing.
* Refer to the notebook for detailed explanations and code implementation of data cleaning, EDA, model loading, LoRA adaptation, training, and evaluation.

### 2. Running the Chatbot Application (Streamlit)
* **Run the application:** To launch the chatbot application, navigate to the project directory and use the following command in the terminal:

```bash 
streamlit run chatbot.py
```
* **Interact with the chatbot:** Enter your queries in the chat input box, and the chatbot will respond based on its understanding of your input and predicted sentiment.

### 3. Installing Required Packages:
* Install the required Python libraries listed in `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```

## **Configuration**:
**Azure Text Analytics:**
   * Set up an Azure Text Analytics resource and obtain the endpoint, key, project name, and deployment name.
## **Project Structure**:

arabic-mental-health-chatbot/
│
├── Arabic_Mental_Health_Chatbot_Fine-tuning.ipynb # Notebook for fine-tuning
│
├── chatbot.py # Streamlit application code
│
├── requirements.txt # Project dependencies 
│
├── .env # Environment variables
│
└── README.md




