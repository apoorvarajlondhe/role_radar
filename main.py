import streamlit as st
import PyPDF2
import textract
import torch
from transformers import DebertaModel,DebertaTokenizer,DebertaConfig
# from transformers import RobertaModel,RobertaTokenizer,RobertaConfig
# from transformers import DistilBertModel,DistilBertTokenizer,DistilBertConfig

# model_path = "/content/drive/MyDrive/Apoorvaraj BE Project/deberta_model.pth"
# model.load_state_dict(torch.load(model_path))

# DEBERTA MODEL
config = DebertaConfig.from_pretrained('microsoft/deberta-base')
model = DebertaModel(config)
model.load_state_dict(torch.load('./deberta_model.pth',map_location=torch.device('cuda')))
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
model.eval()

#ROBERTA MODEL
# config = RobertaConfig.from_pretrained('roberta-base')
# model = RobertaModel(config)
# model.load_state_dict(torch.load('./roberta_model.pth',map_location=torch.device('cpu')))
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model.eval()

#DISTILBERT MODEL
# config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
# model = DistilBertModel(config)
# model.load_state_dict(torch.load('./distilbert_model.pth',map_location=torch.device('cpu')))
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model.eval()



# def predict_score(jd, profile):
#     # Tokenize input text
#     inputs = tokenizer(jd, profile, return_tensors="pt", max_length=512, truncation=True)

#     # Forward pass through the model
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Extract the output embedding
#     output_embedding = outputs.last_hidden_state.mean(dim=1)

#     # Perform further processing or computations as needed
#     # For example, you can pass the output embedding through a linear layer to get the final score
#     # Here, we'll just return the output embedding for demonstration purposes
#     return output_embedding


import torch.nn as nn

class ScoringModel(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(ScoringModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        x = self.relu(x)
        x = self.linear2(x)
        
        x = self.relu(x)
        x = self.linear2(x)
        
        x = self.relu(x)
        x = self.linear3(x)
        
        x = self.sigmoid(x)
        print(x)
        return x
        print(x)

# Initialize the scoring model
scoring_model = ScoringModel(config.hidden_size)

def predict_score(jd, profile):
    # Tokenize input text
    inputs = tokenizer(jd, profile, return_tensors="pt", max_length=512, truncation=True)

    # Forward pass through the BERT model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the output embedding
    output_embedding = outputs.last_hidden_state.mean(dim=1)

    # Forward pass through the scoring model
    score = scoring_model(output_embedding)

    return score.item()





# Function to extract text from PDF file
def extract_text_from_pdf(file):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# # Function to make prediction based on text
# def predict(text):
#     # Your prediction logic here
#     prediction = "Your prediction goes here"
#     return prediction

def main():
    st.title("Role Radar")

    # File upload for first PDF
    st.sidebar.title("Upload JD")
    file_1 = st.sidebar.file_uploader("Choose a PDF file", type="pdf",key='jd')

    # File upload for second PDF
    st.sidebar.title("Upload Resume")
    file_2 = st.sidebar.file_uploader("Choose a PDF file", type="pdf",key='resume')

    if file_1 and file_2:
        if st.sidebar.button("Submit"):
            # Extract text from first PDF
            jd = extract_text_from_pdf(file_1)

            # Extract text from second PDF
            profile = extract_text_from_pdf(file_2)

            # Combine text from both PDFs

            # Make prediction
            prediction = predict_score(jd,profile)
            print("JD:\n",jd)
            print("Resume:\n",profile)
            # Display prediction
            st.success(f"Similarity Score: {prediction*100}%")

if __name__ == "__main__":
    main()
