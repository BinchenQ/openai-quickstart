import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ArgumentParser, ConfigLoader, LOG
from model import GLMModel, OpenAIModel
from translator import PDFTranslator

if __name__ == "__main__":
    argument_parser = ArgumentParser()
    args = argument_parser.parse_arguments()
    config_loader = ConfigLoader(args.config)

    config = config_loader.load_config()

    model_name = args.openai_model if args.openai_model else config['OpenAIModel']['model']
    api_key = args.openai_api_key if args.openai_api_key else config['OpenAIModel']['api_key']
    base_url = args.openai_base_url if args.openai_base_url else config['OpenAIModel']['base_url']
    model = OpenAIModel(model=model_name, api_key=api_key, base_url=base_url)
    books = args.book if args.book else [config['common']['book']]
    file_format = args.file_format if args.file_format else config['common']['file_format']
    translator = PDFTranslator(model)
    for book in books:
        pdf_file_path = book
        if not os.path.exists(pdf_file_path):
            LOG.error(f"File {pdf_file_path} does not exist.")
            continue
        LOG.info(f"Translating {pdf_file_path}...")
        translator.translate_pdf(pdf_file_path, file_format)
