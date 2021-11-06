from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from io import StringIO
import smtplib
import socket

def send_email_notification(subject, message):
    email = 'experimental.tungkvt@gmail.com'
    send_to_email = 'kvttung@gmail.com'
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    subject = subject
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = send_to_email
    msg['Subject'] = subject + ' ' + host_name + ' ' + host_ip
    message = message
    msg.attach(MIMEText(message, 'plain', 'utf-8'))
    # Send the message via SMTP server.
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login('experimental.tungkvt@gmail.com', '123456Aa@')
    text = msg.as_string()
    server.sendmail(email, send_to_email, text)
    server.quit()
    print('email sent to ' + str(send_to_email))
    return True

# try:
#     # Your main code will be placed here
#     print(1+1)
# except Exception as e:
#     log_stream = StringIO()
#     logging.basicConfig(stream=log_stream, level=logging.INFO)
#     logging.error("Exception occurred", exc_info=True)
#     send_email_crash_notification(log_stream.getvalue())

