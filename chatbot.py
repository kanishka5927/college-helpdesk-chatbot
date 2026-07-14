import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_excel("dataset/college_faq.xlsx")

# Load AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

questions = df["Question"].tolist()
embeddings = model.encode(questions)

# Financial amount keywords
financial_keywords = [
    "tuition fee",
    "hostel fee",
    "hostel fees",
    "exam fee",
    "semester fee",
    "admission fee",
    "college fee",
    "fees",
    "fee structure",
    "cost",
    "price",
    "charges",
    "extra fee",
    "extra fees",
    "financial",
    "payment amount",
    "scholarship amount",
    "refund amount"
]

# Words that indicate the user is asking about the PROCESS
process_keywords = [
    "how",
    "where",
    "when",
    "apply",
    "application",
    "pay",
    "payment process",
    "procedure",
    "steps",
    "register",
    "submit"
]


def get_answer(user_question):

    question = user_question.lower()

    # Block only financial amount questions
    if any(word in question for word in financial_keywords):

        # Allow process-related questions
        if not any(word in question for word in process_keywords):
            return (
                "I'm here to help with general college procedures and information. "
                "For fee amounts, financial charges, scholarships, refunds, or any "
                "other financial matters, please contact the College Accounts/Finance Department."
            )

    user_embedding = model.encode([user_question])

    similarity = cosine_similarity(user_embedding, embeddings)

    best_score = similarity.max()
    best_index = similarity.argmax()

    if best_score < 0.50:
        return (
            "Sorry, I couldn't find a relevant answer. "
            "Please contact the college office."
        )

    return df.iloc[best_index]["Answer"]