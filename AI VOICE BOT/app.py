from Aivoicebot import AivoiceAssistant
from email_utils import SendEmail
from Rag_chain import build_qa_chain



def main():
    qa_chain = build_qa_chain()
    assistant = AivoiceAssistant(qa_chain)
    emailer = SendEmail(qa_chain)

    assistant.speak(" Hi! Im AI assistant. Ask me anything ")

    while True:
        command = assistant.listen()

        if assistant.is_goodbye(command):
            assistant.speak("Goodbye!")
            break
        reply = assistant.ask_gpt(command)
        assistant.speak(reply)

        if assistant.detect_email_intent(command):
            assistant.speak("Whom do you want to meet?")
            name = assistant.listen()

            assistant.speak(f"What is the purpose of meeting {name}?")
            purpose = assistant.listen()

            to_email = assistant.extract_email_from_name(name)
            if not to_email:
                assistant.speak("Sorry, I couldn't find their email.")
                continue

            subject, body = emailer.generate_email_subject_and_body(name, purpose)

            assistant.speak("Do you want me to send the appointment email? Say yes or no.")
            confirm = assistant.listen().lower()

            if any(word in confirm.lower() for word in ["yes", "s", "yeah", "ask", "please"]):
                if emailer.send_email(to_email, subject, body):
                    assistant.speak(f"Email has been sent to {name} successfully.")
                else:
                    assistant.speak("I couldn’t send the email.")
            else:
                assistant.speak("Okay, I won't send it.")


if __name__ == "__main__":
    main()
