import pickle
from sklearn.feature_extraction.text import CountVectorizer
import gradio as gr

# Load the Naive Bayes model and vectorizer
with open("naive_bayes_model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define responses dictionary
responses = {
    "activate_my_card": "To activate your card, log in to the app and navigate to the 'Activate Card' section.",
    "age_limit": "The minimum age to use this service is 18 years.",
    "apple_pay_or_google_pay": "Yes, you can use both Apple Pay and Google Pay with your card.",
    "atm_support": "Your card is supported by ATMs that display the Visa or Mastercard logo.",
    "automatic_top_up": "You can enable automatic top-up in the app under the 'Top-Up Settings' section.",
    "balance_not_updated_after_bank_transfer": "If your balance hasn’t updated after a bank transfer, please wait for 24 hours. If the issue persists, contact support.",
    "beneficiary_not_allowed": "Ensure that the beneficiary details are correct. Some accounts may have restrictions; check with customer support for clarification.",
    "cancel_transfer": "To cancel a transfer, go to the 'Transaction History' section in the app and select the transfer you wish to cancel.",
    "card_about_to_expire": "If your card is about to expire, a replacement will be sent automatically. Contact support if you haven’t received it.",
    "card_arrival": "New cards usually arrive within 7-10 business days after being issued.",
    "card_not_working": "If your card isn't working, ensure it is activated and has sufficient balance. Contact support if the issue persists.",
    "change_pin": "You can change your PIN using the app or at any ATM with your card.",
    "contactless_not_working": "Ensure your card supports contactless payments and check if the terminal accepts it. If the issue persists, contact support.",
    "country_support": "Your card is supported in all countries where Visa/Mastercard is accepted. Check the app for restrictions.",
    "declined_card_payment": "Card payments can be declined due to insufficient balance, incorrect PIN, or restrictions on the merchant. Check your app for details.",
    "lost_or_stolen_card": "If your card is lost or stolen, block it immediately in the app under the 'Card Management' section and request a replacement.",
    "pending_card_payment": "Pending payments are usually resolved within 2-3 business days. Contact support if the status doesn't update.",
    "refund_not_showing_up": "Refunds can take up to 7 business days to appear. Check with the merchant or contact support if it takes longer.",
    "top_up_failed": "Top-up failures may occur due to incorrect details or insufficient balance in the source account. Check your app for details.",
    "transfer_not_received_by_recipient": "If the recipient hasn't received the transfer, ensure the details are correct. Contact support for further assistance.",
}

# Define chatbot response logic
def chatbot_response(user_input):
    vectorized_input = vectorizer.transform([user_input])
    predicted_label = clf.predict(vectorized_input)[0]
    return responses.get(predicted_label, "Sorry, I couldn't understand your question.")

# Define Gradio interface
interface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask your fintech question here..."),
    outputs="text",
    title="Banking Chatbot",
    description="A chatbot to handle banking-related queries using a Naive Bayes model."
)

if __name__ == "__main__":
    # Launch the Gradio app
    interface.launch()
