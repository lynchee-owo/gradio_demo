# gradio_demo
import os
import io
import time
import openai
import numpy as np
import gradio as gr
from fpdf import FPDF
from PIL import Image
from dotenv import load_dotenv
from gradio.components import File, Textbox
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

load_dotenv()
azure_key = os.getenv("AZURE_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")
openai_api_key = os.getenv("OPENAI_API_KEY")

client = ComputerVisionClient(azure_endpoint, CognitiveServicesCredentials(azure_key))
openai.api_key = openai_api_key

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', '', 12)
        self.cell(80)
        # self.cell(30, 10, 'Title', 0, 1, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

    def chapter_title(self, num, title):
        self.set_font('Arial', '', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, 'Chapter %s : %s' % (num, title), 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Times', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

def create_pdf(text, filename="output.pdf"):
    pdf = PDF()
    pdf.set_left_margin(20)
    pdf.set_right_margin(20)
    pdf.add_page()
    pdf.chapter_body(text)
    pdf.output(filename)

def process_image(image):
    start_time = time.time()
    # Convert numpy array to PIL Image
    image = Image.fromarray((image * 255).astype(np.uint8))

    # Convert PIL Image to byte stream
    byte_stream = io.BytesIO()
    image.save(byte_stream, format='PNG')
    byte_stream = io.BytesIO(byte_stream.getvalue())

    # Use Azure to perform OCR on the image and extract text
    raw_response = client.read_in_stream(byte_stream, raw=True)

    # Extract operation location from the raw response
    operation_location = raw_response.headers["Operation-Location"]

    # Extract operation id from operation location
    operation_id = operation_location.split("/")[-1]

    # Poll until the operation is finished
    while True:
        result = client.get_read_result(operation_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)
    print("ocr done: ", time.time() - start_time)
    start_time = time.time()

    # Extract the recognized text
    if result.status == OperationStatusCodes.succeeded:
        text = "\n".join([line.text for line in result.analyze_result.read_results[0].lines])
        prompt = "Improve the following class notes to be well-structured and add a title and a summary:\n" + text

        # Use OpenAI to generate a completion based on the recognized text
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        print("openai done: ", time.time() - start_time)
        response_text = response.choices[0].message.content.strip()

        # Create a PDF with the response text
        create_pdf(response_text)

        return response_text, "output.pdf"

iface = gr.Interface(
    fn=process_image,
    inputs="image",
    outputs=[Textbox(label="Response Text"), File(label="Download PDF")],
    interpretation="default",
)
iface.launch(share=True)