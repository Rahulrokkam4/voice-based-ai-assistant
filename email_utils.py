import os
import csv
import json
import smtplib
from datetime import datetime
from email.message import EmailMessage



class SendEmail:
    def __init__(self, qa_chain, log_file="E:\\Datasets\\email_log.csv"):
        self.qa = qa_chain
        self.LOG_FILE = log_file
        self.EMAIL_USER = os.getenv("EMAIL_USER")
        self.EMAIL_PASS = os.getenv("EMAIL_PASS")
        self.EMAIL_HOST = os.getenv("EMAIL_HOST")
        self.EMAIL_PORT = os.getenv("EMAIL_PORT")

    # save email's
    def log_email_row(self, to_email, subject, body):
        try:
            file_exists = os.path.exists(self.LOG_FILE)
            with open(self.LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["timestamp", "to_email", "body"])
                writer.writerow([datetime.now().isoformat(), to_email, subject, body])
        except PermissionError as e:
            print(f"Permission Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

    # generate subject & body from purpose
    def generate_email_subject_and_body(self, name, purpose):
        prompt = f"""
         You are an AI assistant writing professional appointment emails.

         Given that the user wants to book an appointment with {name} for the purpose: "{purpose}", 
         respond with only a JSON object in the following format:
         {{
           "subject": "...",
           "body": "..."
         }}
         Respond ONLY with valid JSON. Do not include explanations or extra text.
         """
        response = self.qa.invoke({"question": prompt})
        result_text = response.get("result", "").strip()
        try:
            data = json.loads(result_text)
            return data["subject"], data["body"]
        except Exception as e:
            print(" Parsing failed:", e)
            return result_text

    # Sending email
    def send_email(self, to_email, subject, body):
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = self.EMAIL_USER
            msg["To"] = to_email
            msg.set_content(body)
            with smtplib.SMTP(self.EMAIL_HOST, 587) as server:
                server.starttls()
                server.login(self.EMAIL_USER, self.EMAIL_PASS)
                server.send_message(msg)
            # save email data
            self.log_email_row(to_email, subject, body)
            return True
        except Exception as e:
            print("Email send failed:", e)
            return False
