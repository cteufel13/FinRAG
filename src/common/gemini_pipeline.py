from google import genai
from langchain.prompts import PromptTemplate
import fitz
import os
from langchain.prompts import PromptTemplate
import json5
import json
import ast
from tqdm import tqdm


class GeminiPipeline:

    def __init__(
        self, prompt_template: PromptTemplate, source_path: str, target_path: str
    ):

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.prompt_template = prompt_template
        self.source_path = source_path
        self.target_path = target_path

    def run(self):
        """Run the pipeline with the given prompt."""

        files = [
            os.path.join(self.source_path, f)
            for f in os.listdir(self.source_path)
            if os.path.isfile(os.path.join(self.source_path, f))
        ]

        for file in tqdm(files):
            print(file)
            if file.endswith(".pdf"):
                # Process the PDF file
                text = self.process_pdf(file)

                # Generate the prompt
                prompt = self.prompt_template.format(document_text=text)

                # Retrieve the content
                flag = False
                while not flag:
                    try:
                        output = self.retrieve(prompt)
                        if isinstance(output, list):
                            output = output[0]
                        # Save the result to the target path
                        output_json = self.post_process_text(output)

                        self.save_text(
                            output_json,
                            os.path.join(
                                self.target_path,
                                f"{os.path.basename(file).replace('.pdf','')}.json",
                            ),
                        )
                        flag = True
                    except Exception as e:
                        print(f"Error: {e}")
                        print("Retrying...")
                        flag = False

    def retrieve(self, prompt: str):
        output = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return output

    def load_pdf(self, pdf_path: str):
        """Load PDF file and return a Document object."""
        doc = fitz.open(pdf_path)

        return doc

    def extract_pdf_text(self, doc, margin_height=50):
        all_text = []

        for page in doc:
            page_height = page.rect.height
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        y0, y1 = span["bbox"][1], span["bbox"][3]

                        # Skip if within header/footer margin
                        if y1 < margin_height or y0 > (page_height - margin_height):
                            continue

                        all_text.append(text)

        return "\n".join(all_text)

    def process_pdf(self, path):
        """Process the PDF file and return the text."""
        doc = self.load_pdf(path)
        text = self.extract_pdf_text(doc)
        return text

    def post_process_text(self, output):
        """Post-process the text to remove unwanted characters."""
        # Remove unwanted characters
        text = output.text.replace("```json", "").replace("```", "")
        clean_json_string = ast.literal_eval(f"'''{text}'''")
        parsed_json = json5.loads(clean_json_string)
        return parsed_json

    def save_text(self, text, path):
        """Save the text to a file."""
        with open(path, "w") as f:
            json.dump(text, f, indent=4)
