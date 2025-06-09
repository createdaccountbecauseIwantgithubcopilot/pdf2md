#!/usr/bin/env python3
"""
PDF to Images Converter and Transcriber
Converts a PDF file to images and either packages them in a zip file or transcribes to markdown.
"""

import os
import sys
import zipfile
from pathlib import Path
from pdf2image import convert_from_path
import argparse
from google import genai
from google.genai import types
from PIL import Image


def get_output_mode():
    """Prompt user to select output mode."""
    print("\nSelect output mode:")
    print("1. Create ZIP file with images")
    print("2. Transcribe to Markdown")
    
    while True:
        choice = input("Enter your choice (1-2): ").strip()
        if choice in ["1", "2"]:
            return choice
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_quality_preset():
    """Prompt user to select quality preset."""
    print("\nSelect quality preset:")
    print("1. Low (150 DPI)")
    print("2. Medium (200 DPI)")
    print("3. High (300 DPI)")
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice == "1":
            return 150
        elif choice == "2":
            return 200
        elif choice == "3":
            return 300
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def check_existing_files(pdf_path, output_mode):
    """Check if output files already exist and prompt user for action."""
    pdf_name = Path(pdf_path).stem
    
    # Determine which file to check based on output mode
    if output_mode == "1":
        output_file = f"{pdf_name}.zip"
        file_type = "ZIP"
    else:
        output_file = f"{pdf_name}.md"
        file_type = "Markdown"
    
    # Check if file exists
    if os.path.exists(output_file):
        print(f"\nWarning: {file_type} file '{output_file}' already exists!")
        print("What would you like to do?")
        print("1. Delete the existing file and continue")
        print("2. Exit without making changes")
        
        while True:
            choice = input("Enter your choice (1-2): ").strip()
            if choice == "1":
                try:
                    os.remove(output_file)
                    print(f"Deleted existing file: {output_file}")
                    return True
                except Exception as e:
                    print(f"Error deleting file: {e}")
                    sys.exit(1)
            elif choice == "2":
                print("Exiting without making changes.")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    return True


def convert_pdf_to_images(pdf_path, dpi):
    """Convert PDF pages to images."""
    print(f"\nConverting PDF pages to images at {dpi} DPI...")
    
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Successfully converted {len(images)} pages")
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        sys.exit(1)


def create_zip_file(pdf_path, images, dpi):
    """Create a zip file containing the images."""
    pdf_name = Path(pdf_path).stem
    zip_filename = f"{pdf_name}.zip"
    folder_name = pdf_name
    
    print(f"\nCreating zip file: {zip_filename}")
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, image in enumerate(images, 1):
                # Create image filename
                image_filename = f"page_{i:03d}.png"
                image_path = os.path.join(folder_name, image_filename)
                
                # Save image to temporary file
                temp_filename = f"temp_page_{i}.png"
                image.save(temp_filename, 'PNG')
                
                # Add to zip with proper path
                zipf.write(temp_filename, image_path)
                
                # Remove temporary file
                os.remove(temp_filename)
                
                print(f"  Added page {i}/{len(images)}")
        
        print(f"\nSuccessfully created {zip_filename}")
        print(f"Total size: {os.path.getsize(zip_filename) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"Error creating zip file: {e}")
        sys.exit(1)


def setup_gemini_client():
    """Setup Google Gemini client."""
    return genai.Client()


def transcribe_image_to_markdown(client:genai.Client, image, page_num, total_pages):
    """Transcribe a single image to markdown using Gemini API."""
    print(f"  Transcribing page {page_num}/{total_pages}...")
    
    # Craft a detailed prompt for accurate transcription
    prompt = """Please transcribe ALL text content from this image into properly formatted Markdown.
    
Important instructions:
    1. Preserve the exact text content - do not summarize or paraphrase
    2. Use appropriate Markdown formatting:
       - Use # for main headings, ## for subheadings, etc.
       - Use **bold** for emphasized text
       - Use *italic* for italicized text
       - Use bullet points or numbered lists where appropriate
    3. For mathematical expressions and formulas:
       - Use LaTeX notation enclosed in $ for inline math
       - Use $$ for display math equations
       - Ensure all mathematical symbols are properly converted to LaTeX
    4. For footnotes:
       - Clearly mark them as [^footnote_number] in the main text
       - List footnote content at the end with [^footnote_number]: footnote text
    5. For tables, use proper Markdown table syntax
    6. Preserve paragraph breaks and text structure
    7. If there are any figures or diagrams, add a descriptive note like: [Figure: description]
    
    Transcribe the complete page content now:"""
    
    try:
        # Generate content with the PIL image directly
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-04-17',
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                system_instruction="You are a document transcriber who makes no mistakes",
                temperature=0.0
            )
        )
        
        return response.text
    except Exception as e:
        print(f"    Error transcribing page {page_num}: {e}")
        return f"\n[Error transcribing page {page_num}]\n"


def create_markdown_file(pdf_path, images):
    """Transcribe all images and create a markdown file."""
    pdf_name = Path(pdf_path).stem
    markdown_filename = f"{pdf_name}.md"
    
    print(f"\nSetting up Gemini API...")
    client = setup_gemini_client()
    
    print(f"\nTranscribing {len(images)} pages to markdown...")
    
    try:
        with open(markdown_filename, 'w', encoding='utf-8') as md_file:
            # Add header
            md_file.write(f"# {pdf_name}\n\n")
            md_file.write(f"*Transcribed from PDF with {len(images)} pages*\n\n")
            md_file.write("---\n\n")
            
            # Transcribe each page
            for i, image in enumerate(images, 1):
                # Add page separator
                if i > 1:
                    md_file.write("\n\n---\n\n")
                
                md_file.write(f"## Page {i}\n\n")
                
                # Transcribe the page
                transcribed_text = transcribe_image_to_markdown(client, image, i, len(images))
                md_file.write(transcribed_text)
                md_file.write("\n")
        
        print(f"\nSuccessfully created {markdown_filename}")
        print(f"File size: {os.path.getsize(markdown_filename) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Error creating markdown file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to images (zip) or transcribe to markdown")
    parser.add_argument("pdf_file", help="Name of the PDF file to convert")
    
    args = parser.parse_args()
    pdf_file = args.pdf_file
    
    # Check if file exists
    if not os.path.exists(pdf_file):
        print(f"Error: File '{pdf_file}' not found in current directory")
        sys.exit(1)
    
    # Check if it's a PDF file
    if not pdf_file.lower().endswith('.pdf'):
        print(f"Error: File '{pdf_file}' is not a PDF file")
        sys.exit(1)
    
    print(f"Processing PDF file: {pdf_file}")
    
    # Get output mode from user
    output_mode = get_output_mode()
    
    # Get quality preset from user
    dpi = get_quality_preset()
    
    # Check for existing output files
    check_existing_files(pdf_file, output_mode)
    
    # Convert PDF to images
    images = convert_pdf_to_images(pdf_file, dpi)
    
    # Process based on selected mode
    if output_mode == "1":
        # Create zip file
        create_zip_file(pdf_file, images, dpi)
    else:
        # Transcribe to markdown
        create_markdown_file(pdf_file, images)
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()