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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
import typing_extensions as typing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from functools import partial

api_key = os.environ.get('GOOGLE_API_KEY')
print(f"Running with API key: {api_key}")

# Structured output for quality verification
class TranscriptionQuality(typing.TypedDict):
    is_good_quality: bool
    feedback: str

# Cost tracking
cost_tracker = {
    'transcription_tokens': {'thoughts': 0, 'output': 0},
    'verification_tokens': {'thoughts': 0, 'output': 0},
    'total_requests': 0,
    'retry_count': 0
}

# Thread-safe lock for cost tracking
cost_lock = threading.Lock()

# Page status tracking for concurrent processing
page_status = {}


def get_output_mode(mode_arg=None):
    """Get output mode from command line argument or prompt user."""
    # If mode was provided via command line, use it
    if mode_arg is not None:
        if mode_arg in ["1", "2"]:
            return mode_arg
        else:
            print(f"Error: Invalid mode '{mode_arg}'. Mode must be 1 or 2.")
            sys.exit(1)
    
    # Otherwise, prompt user interactively
    print("\nSelect output mode:")
    print("1. Create ZIP file with images")
    print("2. Transcribe to Markdown")
    
    while True:
        choice = input("Enter your choice (1-2): ").strip()
        if choice in ["1", "2"]:
            return choice
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_quality_preset(quality_arg=None):
    """Get quality preset from command line argument or prompt user."""
    # If quality was provided via command line, use it
    if quality_arg is not None:
        if quality_arg == "1":
            return 150
        elif quality_arg == "2":
            return 200
        elif quality_arg == "3":
            return 300
        else:
            print(f"Error: Invalid quality '{quality_arg}'. Quality must be 1, 2, or 3.")
            sys.exit(1)
    
    # Otherwise, prompt user interactively
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


def check_existing_files(pdf_path, output_mode, overwrite=False):
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
        if overwrite:
            # Automatically delete the existing file
            try:
                os.remove(output_file)
                print(f"Overwriting existing file: {output_file}")
                return True
            except Exception as e:
                print(f"Error deleting file: {e}")
                sys.exit(1)
        else:
            # Exit with a message
            print(f"Error: {file_type} file '{output_file}' already exists!")
            print("Use --overwrite flag to automatically overwrite existing files.")
            sys.exit(1)
    
    return True


def convert_pdf_to_images(pdf_path, dpi):
    """Convert PDF pages to images."""
    console = Console()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"[cyan]Converting PDF to images at {dpi} DPI...", total=None)
        
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            progress.update(task, description=f"[green]Successfully converted {len(images)} pages")
            return images
        except Exception as e:
            console.print(f"[bold red]Error converting PDF:[/bold red] {e}")
            sys.exit(1)


def create_zip_file(pdf_path, images, dpi):
    """Create a zip file containing the images."""
    pdf_name = Path(pdf_path).stem
    zip_filename = f"{pdf_name}.zip"
    folder_name = pdf_name
    
    console = Console()
    console.print(f"\n[bold cyan]Creating zip file:[/bold cyan] {zip_filename}")
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]Adding pages to ZIP...", total=len(images))
                
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
                    
                    progress.update(task, advance=1, description=f"[cyan]Adding page {i}/{len(images)}...")
        
        console.print(f"\n[bold green]✓[/bold green] Successfully created {zip_filename}")
        console.print(f"[dim]Total size: {os.path.getsize(zip_filename) / 1024 / 1024:.2f} MB[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error creating zip file:[/bold red] {e}")
        sys.exit(1)


def setup_gemini_client():
    """Setup Google Gemini client."""
    return genai.Client(api_key=api_key)


def verify_transcription(client: genai.Client, image, transcription, original_prompt, feedback_history=""):
    """Verify the quality of a transcription using structured output."""
    verification_prompt = f"""You are a quality assurance checker reviewing a document transcription. 
    
Original transcription prompt:
{original_prompt}

Transcription result:
{transcription}

{feedback_history}

Compare the transcription with the source image with the following STRICT criteria:

1. Mathematical Formulas - STRICT (100% accuracy required):
   - ALL mathematical expressions, equations, and formulas must be EXACTLY correct
   - Check every symbol, subscript, superscript, and operator
   - Verify LaTeX syntax is correct and will render properly
   - Even minor errors in formulas are NOT acceptable

2. Important Content/Information - STRICT:
   - All key facts, data, numbers, names, and technical terms must be 100% accurate
   - No paraphrasing or summarization of important content
   - Tables must contain all data exactly as shown
   - Citations and references must be complete and accurate

3. Styling and Formatting - LENIENT:
   - Minor variations in Markdown formatting are acceptable
   - Paragraph breaks and spacing can vary slightly
   - Bold/italic emphasis can be interpreted reasonably
   - List formatting (bullets vs numbers) can vary if content is preserved

Mark as needing improvement if:
- ANY mathematical formula has errors (even minor ones)
- ANY important information is missing or incorrect
- Major structural elements are missing

Provide your assessment as a boolean (true if meets all criteria, false if any issues) and specific feedback."""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-05-20',
            contents=[verification_prompt, image],
            config=types.GenerateContentConfig(
                system_instruction="You are a quality assurance specialist who evaluates document transcriptions for accuracy and completeness.",
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=TranscriptionQuality,
                thinking_config=types.ThinkingConfig(thinking_budget=24576)
            )
        )
        
        # Track token usage (thread-safe)
        with cost_lock:
            if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
                cost_tracker['verification_tokens']['thoughts'] += getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0
                cost_tracker['verification_tokens']['output'] += getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
            else:
                # If usage_metadata is None, just increment by 0
                cost_tracker['verification_tokens']['thoughts'] += 0
                cost_tracker['verification_tokens']['output'] += 0
        
        result = response.parsed if hasattr(response, 'parsed') else None
        return result if result else {"is_good_quality": True, "feedback": "Verification result parsing failed, accepting as-is"}
    except Exception as e:
        return {"is_good_quality": True, "feedback": f"Verification failed ({str(e)[:50]}...), accepting as-is"}


def transcribe_page_concurrent(client: genai.Client, image, page_num, status_callback=None):
    """Transcribe a single page for concurrent processing with status updates."""
    base_prompt = """Please transcribe ALL text content from this image into properly formatted Markdown.
    
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
    
    max_retries = 10
    retry_count = 0
    feedback_history = ""
    current_prompt = base_prompt
    
    # Update status
    if status_callback:
        status_callback(page_num, "transcribing", "")
    
    while retry_count <= max_retries:
        try:
            # Generate transcription
            response = client.models.generate_content(
                model='gemini-2.5-flash-preview-05-20',
                contents=[current_prompt, image],
                config=types.GenerateContentConfig(
                    system_instruction="You are a document transcriber who is given images and then transcribes them into markdown documents following a strict format.",
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=24576)
                )
            )
            
            # Track token usage (thread-safe)
            with cost_lock:
                if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
                    cost_tracker['transcription_tokens']['thoughts'] += getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0
                    cost_tracker['transcription_tokens']['output'] += getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
                else:
                    # If usage_metadata is None, just increment by 0
                    cost_tracker['transcription_tokens']['thoughts'] += 0
                    cost_tracker['transcription_tokens']['output'] += 0
                cost_tracker['total_requests'] += 1
            
            transcription = response.text
            
            # Update status for verification
            if status_callback:
                status_callback(page_num, "verifying", "")
            
            # Verify transcription quality
            verification = verify_transcription(client, image, transcription, base_prompt, feedback_history)
            
            with cost_lock:
                cost_tracker['total_requests'] += 1
            
            if verification['is_good_quality']:
                # Good quality, return the transcription
                if status_callback:
                    status_callback(page_num, "completed", "")
                return (page_num, transcription, None)
            else:
                # Need to retry with feedback
                retry_count += 1
                with cost_lock:
                    cost_tracker['retry_count'] += 1
                
                if retry_count <= max_retries:
                    feedback_history += f"\n\nPrevious attempt {retry_count} feedback:\n{verification['feedback']}"
                    current_prompt = f"{base_prompt}\n\nPlease address the following issues from the previous transcription attempts:{feedback_history}"
                    if status_callback:
                        status_callback(page_num, f"retry_{retry_count}", "")
                else:
                    if status_callback:
                        status_callback(page_num, "completed_with_issues", "Max retries reached")
                    # Return with feedback history for debug file
                    return (page_num, transcription, {"error": "Max retries reached", "feedback_history": feedback_history})
                    
        except Exception as e:
            if status_callback:
                status_callback(page_num, "error", str(e)[:40])
            return (page_num, f"\n[Error transcribing page {page_num}: {e}]\n", str(e))
    
    return (page_num, f"\n[Error: Max retries exceeded for page {page_num}]\n", "Max retries exceeded")


def transcribe_image_to_markdown(client: genai.Client, image, page_num, total_pages, progress=None, task=None):
    """Transcribe a single image to markdown using Gemini API with quality verification."""
    base_prompt = """Please transcribe ALL text content from this image into properly formatted Markdown.
    
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
    
    console = Console()
    max_retries = 10
    retry_count = 0
    feedback_history = ""
    current_prompt = base_prompt
    
    while retry_count <= max_retries:
        try:
            # Update progress for retries
            if progress and task and retry_count > 0:
                progress.update(task, description=f"[yellow]Re-transcribing page {page_num}/{total_pages} (attempt {retry_count + 1})...")
            
            # Generate transcription
            response = client.models.generate_content(
                model='gemini-2.5-flash-preview-05-20',
                contents=[current_prompt, image],
                config=types.GenerateContentConfig(
                    system_instruction="You are a document transcriber who is given images and then transcribes them into markdown documents following a strict format.",
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_budget=24576)
                )
            )
            
            # Track token usage (thread-safe)
            with cost_lock:
                if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
                    cost_tracker['transcription_tokens']['thoughts'] += getattr(response.usage_metadata, 'thoughts_token_count', 0) or 0
                    cost_tracker['transcription_tokens']['output'] += getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
                else:
                    # If usage_metadata is None, just increment by 0
                    cost_tracker['transcription_tokens']['thoughts'] += 0
                    cost_tracker['transcription_tokens']['output'] += 0
                cost_tracker['total_requests'] += 1
            
            transcription = response.text
            
            # Verify transcription quality
            if progress and task:
                progress.update(task, description=f"[blue]Verifying page {page_num}/{total_pages}...")
            
            verification = verify_transcription(client, image, transcription, base_prompt, feedback_history)
            with cost_lock:
                cost_tracker['total_requests'] += 1
            
            if verification['is_good_quality']:
                # Good quality, return the transcription
                return transcription
            else:
                # Need to retry with feedback
                retry_count += 1
                with cost_lock:
                    cost_tracker['retry_count'] += 1
                
                if retry_count <= max_retries:
                    feedback_history += f"\n\nPrevious attempt {retry_count} feedback:\n{verification['feedback']}"
                    current_prompt = f"{base_prompt}\n\nPlease address the following issues from the previous transcription attempts:{feedback_history}"
                    if progress and task:
                        progress.update(task, description=f"[yellow]Re-transcribing page {page_num} - {verification['feedback'][:50]}...[/yellow]")
                else:
                    if progress and task:
                        progress.update(task, description=f"[yellow]Page {page_num} - Max retries reached, using last version[/yellow]")
                    return transcription
                    
        except Exception as e:
            if progress and task:
                progress.update(task, description=f"[red]Error on page {page_num}: {str(e)[:50]}...[/red]")
            return f"\n[Error transcribing page {page_num}: {e}]\n"
    
    return f"\n[Error: Max retries exceeded for page {page_num}]\n"


def create_progress_display(total_pages):
    """Create a rich progress display for concurrent transcription."""
    console = Console()
    
    # Initialize page status for all pages
    for i in range(1, total_pages + 1):
        page_status[i] = ("[dim]Waiting[/dim]", "")
    
    # Create overall progress
    overall_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    overall_task = overall_progress.add_task("[cyan]Transcribing pages...", total=total_pages)
    
    def create_status_table():
        """Create a fresh table with current status."""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Page", style="dim", width=6)
        table.add_column("Status", width=20)
        table.add_column("Details", width=40)
        
        # Add rows with current status, excluding completed pages
        for i in range(1, total_pages + 1):
            status, details = page_status.get(i, ("[dim]Waiting[/dim]", ""))
            # Skip completed pages to keep the table focused on active work
            if "✓ Completed" in status:
                continue
            # Ensure all values are strings
            table.add_row(str(i), str(status), str(details))
        
        return table
    
    # Combine into panel
    def create_panel():
        main_layout = Layout()
        main_layout.split_column(
            Layout(overall_progress, size=3),
            Layout(create_status_table())
        )
        
        return Panel(
            main_layout,
            title="[bold blue]Concurrent Transcription Progress[/bold blue]",
            border_style="blue"
        )
    
    return console, create_panel, create_status_table, overall_progress, overall_task


def update_page_status(page_num, status, progress_table, details=""):
    """Update the status of a specific page in the progress table."""
    status_colors = {
        "waiting": "[dim]Waiting[/dim]",
        "transcribing": "[yellow]Transcribing...[/yellow]",
        "verifying": "[blue]Verifying...[/blue]",
        "retry_1": "[orange1]Retry 1...[/orange1]",
        "retry_2": "[orange3]Retry 2...[/orange3]",
        "retry_3": "[red]Retry 3...[/red]",
        "completed": "[green]✓ Completed[/green]",
        "completed_with_issues": "[yellow]⚠ Completed (with issues)[/yellow]",
        "error": "[red]✗ Error[/red]"
    }
    
    # Handle retry statuses dynamically
    if status.startswith("retry_") and status not in status_colors:
        retry_num = status.split("_")[1]
        if retry_num.isdigit():
            retry_count = int(retry_num)
            if retry_count <= 3:
                # Use the predefined colors
                status_display = status_colors.get(status, str(status))
            else:
                # For retries > 3, use consistent formatting
                status_display = f"[bold red]Retry {retry_count}...[/bold red]"
        else:
            status_display = str(status)
    else:
        status_display = status_colors.get(status, str(status))
    
    # Ensure details is a string
    details_str = str(details) if details is not None else ""
    
    # Store the page status globally for updating the table
    # Safely slice the details string
    truncated_details = details_str[:40] if details_str else ""
    page_status[page_num] = (status_display, truncated_details)


def create_markdown_file_concurrent(pdf_path, images):
    """Transcribe all images concurrently and create a markdown file."""
    pdf_name = Path(pdf_path).stem
    markdown_filename = f"{pdf_name}.md"
    
    console = Console()
    console.print(f"\n[bold cyan]Setting up Gemini API...[/bold cyan]")
    client = setup_gemini_client()
    
    console.print(f"\n[bold cyan]Starting concurrent transcription of {len(images)} pages...[/bold cyan]")
    
    # Set up progress display
    _, create_panel, create_status_table, overall_progress, overall_task = create_progress_display(len(images))
    
    # Thread-safe status update function
    status_lock = threading.Lock()
    completed_pages = 0
    live_display = None  # Will be set inside the with Live block
    
    def status_callback(page_num, status, details=""):
        nonlocal completed_pages
        with status_lock:
            update_page_status(page_num, status, None, details)
            if status in ["completed", "completed_with_issues", "error"]:
                completed_pages += 1
                overall_progress.update(overall_task, advance=1)
        # Update the live display (thread-safe)
        if live_display:
            try:
                live_display.update(create_panel())
            except:
                pass  # Ignore any display update errors
    
    try:
        # Prepare results dictionary
        results = {}
        # Track pages that hit retry limit for debug file
        debug_pages = {}
        
        # Use ThreadPoolExecutor for concurrent transcription
        # WARNING: Using all pages concurrently may hit API rate limits for large PDFs
        max_workers = len(images)  # Process all pages concurrently
        
        with Live(create_panel(), console=console, refresh_per_second=30) as live:
            live_display = live  # Set the reference for the callback
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all transcription tasks
                future_to_page = {}
                for i, image in enumerate(images, 1):
                    future = executor.submit(
                        transcribe_page_concurrent, 
                        client, 
                        image, 
                        i,
                        status_callback
                    )
                    future_to_page[future] = i
                
                # Process completed futures
                for future in as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        result = future.result()
                        if result and len(result) >= 3:
                            result_page_num, transcription, error = result
                            results[result_page_num] = transcription
                            if error:
                                if isinstance(error, dict) and error.get("error") == "Max retries reached":
                                    # Store debug info for pages that hit retry limit
                                    debug_pages[result_page_num] = {
                                        "transcription": transcription,
                                        "feedback_history": error.get("feedback_history", "")
                                    }
                                    console.print(f"[yellow]Page {result_page_num} completed with issues: Max retries reached[/yellow]")
                                else:
                                    console.print(f"[yellow]Page {result_page_num} completed with issues: {error}[/yellow]")
                        else:
                            results[page_num] = f"\n[Error processing page {page_num}: Invalid result format]\n"
                            console.print(f"[red]Page {page_num} returned invalid result format[/red]")
                    except Exception as exc:
                        results[page_num] = f"\n[Error processing page {page_num}: {exc}]\n"
                        console.print(f"[red]Page {page_num} generated an exception: {exc}[/red]")
                    # Update display after each future completes
                    live.update(create_panel())
        
        # Write results to file in order
        console.print(f"\n[bold cyan]Writing markdown file...[/bold cyan]")
        
        with open(markdown_filename, 'w', encoding='utf-8') as md_file:
            # Add header
            md_file.write(f"# {pdf_name}\n\n")
            md_file.write(f"*Transcribed from PDF with {len(images)} pages*\n\n")
            md_file.write("---\n\n")
            
            # Write pages in order
            for i in range(1, len(images) + 1):
                if i > 1:
                    md_file.write("\n\n---\n\n")
                
                md_file.write(f"## Page {i}\n\n")
                
                if i in results:
                    md_file.write(results[i])
                else:
                    md_file.write(f"\n[Error: Page {i} missing from results]\n")
                
                md_file.write("\n")
        
        console.print(f"\n[bold green]✓[/bold green] Successfully created {markdown_filename}")
        console.print(f"[dim]File size: {os.path.getsize(markdown_filename) / 1024:.2f} KB[/dim]")
        
        # Create debug file if any pages hit retry limit
        if debug_pages:
            debug_filename = f"{pdf_name}_error_debug.md"
            console.print(f"\n[bold yellow]Creating debug file for {len(debug_pages)} pages that hit retry limit...[/bold yellow]")
            
            with open(debug_filename, 'w', encoding='utf-8') as debug_file:
                debug_file.write(f"# {pdf_name} - Error Debug Report\n\n")
                debug_file.write(f"*This file contains debug information for pages that reached the retry limit*\n\n")
                debug_file.write(f"Total pages with max retries: {len(debug_pages)}\n\n")
                debug_file.write("---\n\n")
                
                for page_num in sorted(debug_pages.keys()):
                    debug_info = debug_pages[page_num]
                    debug_file.write(f"## Page {page_num}\n\n")
                    debug_file.write("### Final Transcription:\n\n")
                    debug_file.write(debug_info['transcription'])
                    debug_file.write("\n\n### Verification Feedback History:\n")
                    debug_file.write(debug_info['feedback_history'])
                    debug_file.write("\n\n---\n\n")
            
            console.print(f"[bold yellow]✓[/bold yellow] Created debug file: {debug_filename}")
            console.print(f"[dim]Debug file size: {os.path.getsize(debug_filename) / 1024:.2f} KB[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error creating markdown file:[/bold red] {e}")
        sys.exit(1)


def create_markdown_file(pdf_path, images):
    """Transcribe all images and create a markdown file."""
    pdf_name = Path(pdf_path).stem
    markdown_filename = f"{pdf_name}.md"
    
    console = Console()
    console.print(f"\n[bold cyan]Setting up Gemini API...[/bold cyan]")
    client = setup_gemini_client()
    
    console.print(f"\n[bold cyan]Transcribing {len(images)} pages to markdown...[/bold cyan]")
    
    try:
        with open(markdown_filename, 'w', encoding='utf-8') as md_file:
            # Add header
            md_file.write(f"# {pdf_name}\n\n")
            md_file.write(f"*Transcribed from PDF with {len(images)} pages*\n\n")
            md_file.write("---\n\n")
            
            # Transcribe each page with progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]Transcribing pages...", total=len(images))
                
                for i, image in enumerate(images, 1):
                    # Update progress description
                    progress.update(task, description=f"[cyan]Transcribing page {i}/{len(images)}...")
                    
                    # Add page separator
                    if i > 1:
                        md_file.write("\n\n---\n\n")
                    
                    md_file.write(f"## Page {i}\n\n")
                    
                    # Transcribe the page with verification
                    transcribed_text = transcribe_image_to_markdown(client, image, i, len(images), progress, task)
                    md_file.write(transcribed_text)
                    md_file.write("\n")
                    
                    # Update progress
                    progress.update(task, advance=1)
        
        console.print(f"\n[bold green]✓[/bold green] Successfully created {markdown_filename}")
        console.print(f"[dim]File size: {os.path.getsize(markdown_filename) / 1024:.2f} KB[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error creating markdown file:[/bold red] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to images (zip) or transcribe to markdown")
    parser.add_argument("pdf_file", help="Name of the PDF file to convert")
    parser.add_argument("--mode", choices=["1", "2"], help="Output mode: 1 for ZIP file, 2 for Markdown transcription")
    parser.add_argument("--quality", choices=["1", "2", "3"], help="Quality preset: 1 for Low (150 DPI), 2 for Medium (200 DPI), 3 for High (300 DPI)")
    parser.add_argument("--overwrite", action="store_true", help="Automatically overwrite existing output files without prompting")
    
    args = parser.parse_args()
    pdf_file = args.pdf_file
    
    console = Console()
    
    # Check if file exists
    if not os.path.exists(pdf_file):
        console.print(f"[bold red]Error:[/bold red] File '{pdf_file}' not found in current directory")
        sys.exit(1)
    
    # Check if it's a PDF file
    if not pdf_file.lower().endswith('.pdf'):
        console.print(f"[bold red]Error:[/bold red] File '{pdf_file}' is not a PDF file")
        sys.exit(1)
    
    console.print(f"[bold cyan]Processing PDF file:[/bold cyan] {pdf_file}")
    
    # Get output mode from command line or user
    output_mode = get_output_mode(args.mode)
    
    # Get quality preset from command line or user
    dpi = get_quality_preset(args.quality)
    
    # Check for existing output files
    check_existing_files(pdf_file, output_mode, args.overwrite)
    
    # Convert PDF to images
    images = convert_pdf_to_images(pdf_file, dpi)
    
    # Process based on selected mode
    if output_mode == "1":
        # Create zip file
        create_zip_file(pdf_file, images, dpi)
    else:
        # Transcribe to markdown with concurrent processing
        create_markdown_file_concurrent(pdf_file, images)
    
    # Print cost summary if in transcription mode
    if output_mode == "2":
        console.print("\n[bold cyan]Cost Summary:[/bold cyan]")
        console.print(f"[dim]Total API requests: {cost_tracker['total_requests']}[/dim]")
        console.print(f"[dim]Pages requiring retry: {cost_tracker['retry_count']}[/dim]")
        console.print(f"[dim]Transcription tokens - Thoughts: {cost_tracker['transcription_tokens']['thoughts']:,}[/dim]")
        console.print(f"[dim]Transcription tokens - Output: {cost_tracker['transcription_tokens']['output']:,}[/dim]")
        console.print(f"[dim]Verification tokens - Thoughts: {cost_tracker['verification_tokens']['thoughts']:,}[/dim]")
        console.print(f"[dim]Verification tokens - Output: {cost_tracker['verification_tokens']['output']:,}[/dim]")
        total_thoughts = cost_tracker['transcription_tokens']['thoughts'] + cost_tracker['verification_tokens']['thoughts']
        total_output = cost_tracker['transcription_tokens']['output'] + cost_tracker['verification_tokens']['output']
        console.print(f"[bold]Total tokens - Thoughts: {total_thoughts:,}, Output: {total_output:,}[/bold]")
    
    console.print("\n[bold green]Process complete! ✨[/bold green]")


if __name__ == "__main__":
    main()
