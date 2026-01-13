import requests
import os
import json
from PIL import Image
import fitz  # PyMuPDF
import io
from dotenv import load_dotenv
from google import genai
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

# ---------------- LOAD ENV ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# assert GEMINI_API_KEY, "❌ GEMINI_API_KEY not found in .env"

# ---------------- CONFIG ------------------
OUTPUT_DIR = "gemini_ocr_output3"
MODEL_NAME = "gemini-3-flash-preview"
# MODEL_NAME = "gemini-2.5-pro"
# MODEL_NAME = "gemini-3-pro-preview"
# MODEL_NAME = "gemini-3-pro-image-preview"


# ---------------- LOGGING CONFIG ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ocr_app.log')
    ]
)
logger = logging.getLogger("ocr_app")

# ---------------- INIT GEMINI (NEW SDK) ---
# client = genai.Client(api_key=GEMINI_API_KEY)
# Initialize client only if key is present to avoid errors if env not set yet
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    logger.warning("GEMINI_API_KEY not found. OCR will fail.")


def split_pdf_to_images(pdf_url, save_dir=None):
    """
    Downloads a PDF from a URL, splits it into images, 
    and optionally saves them to a directory.
    
    Args:
        pdf_url (str): The URL of the PDF.
        save_dir (str, optional): Directory to save images.
        
    Returns:
        list: List of PIL Image objects.
    """
    logger.info(f"Downloading PDF from: {pdf_url}")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        logger.info("PDF downloaded, opening with PyMuPDF...")
        pdf_document = fitz.open(stream=response.content, filetype="pdf")
        images = []
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            
            if save_dir:
                image_path = os.path.join(save_dir, f"page_{page_num+1}.jpg")
                img.save(image_path, "JPEG")
                logger.info(f"Saved image to {image_path}")
        
        logger.info(f"Converted PDF to {len(images)} images.")
        return images
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return []


UNIVERSAL_SYSTEM_INSTRUCTION = "You are an expert Document Intelligence AI specialized in Andhra Pradesh land revenue records.You are proficient in reading handwritten Telugu and English script. You must infer meaning from handwritten tabular data even when column headers are missing."
UNIVERSAL_PROMPT = """
Objective:
Identify and extract ONLY the Survey Numbers from the document.

Important:
• Survey number usually present in column 2


How to identify page number:
- Page number usually appears at the top of the page.
- page number may be in the format of 1 of 10 or 1 of 100 or 1 of 1000 etc or just number 1,2,3 etc. 
- if page number is not present, return empty or blank " ".
- if page number is format x of y , return only x.
- do not consider other numbers like grama lekh number, survey number etc for page number.

How to identify Survey Numbers:
Survey Numbers usually:
- Are short numeric or alphanumeric values.
- Appear like: 12, 12/1, 28-3, 13-A, 424/3, 215/3, 17-A, 105-D, 183 (10).
- Appear repeatedly in the same vertical column.
- Do NOT contain decimals like 1.25 or 0.75.
- Do NOT contain currency symbols.
- Do NOT look like names or sentences.
- Survey number column is the second column from left side of the document.
- If column headers (e.g., "Survey No", "S.No", "సర్వే నంబరు") are present, use that column; otherwise, identify the column based on the data format.


Handling special cases:
- identify the number of rows in the document. each row has only one survey number only.
- Identify the survey number column as the column where ALL rows follow the SAME FORMAT pattern (e.g., number, number/number, number-letter, number-number).
- Ignore numeric values that follow different formats in different rows.
- Ignore totals, remarks, and assessment rows.

Output Rules:
• Return ONLY JSON
• Do NOT guess missing values
• Preserve exact formatting as written (e.g., 28-3, 13-A)

JSON Output Format:
{
  "page_no": "1",
  "survey_numbers": [
    "28-3",
    "26-1",
    "13-A"
  ]
}
"""



# ---------------- OCR FUNCTION ------------
def gemini_ocr(image, page_no, prompt_type="unknown"):
    logger.info(f"Starting OCR for page {page_no} with prompt_type='{prompt_type}'")
    
    if not client:
        logger.error("Gemini client not initialized.")
        return ""

    # Preprocessing: Convert to Grayscale & Resize
    try:
        image = image.convert("L")  # Convert to Grayscale
        image.thumbnail((1600, 1600)) # Resize to max 1600x1600 maintaining aspect ratio
        
        # Optimize: JPEG Quality 70
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        buffer.seek(0)
        image = Image.open(buffer)
        
        logger.info(f"Image preprocessed: Grayscale, Resized & Compressed (Quality=70) to {image.size}")
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")

    # Use only the UNIVERSAL prompt for this script as requested
    if prompt_type == "unknown":
        system_instruction = UNIVERSAL_SYSTEM_INSTRUCTION
        prompt = UNIVERSAL_PROMPT
    else:
         # Fallback or other types can be added here if needed, 
         # but for now we default to unknown/universal as per the snippet in file
        system_instruction = UNIVERSAL_SYSTEM_INSTRUCTION
        prompt = UNIVERSAL_PROMPT

    logger.debug("Prompt selected. Sending request to Gemini...")
    safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image],
            config={
                'temperature': 0.1,
                'response_mime_type': 'application/json',
                'system_instruction': system_instruction,
                'max_output_tokens': 65536,
                'top_k': 1,
                'top_p': 0.9,
                'safety_settings': safety_settings
            }
        )
        logger.info(f"Gemini response for page {page_no}: {response.text}")
        return response.text or ""
    except Exception as e:
        logger.error(f"Error during Gemini OCR for page {page_no}: {str(e)}")
        return ""


# Thread pool for running sync OCR in async context
_ocr_executor = ThreadPoolExecutor(max_workers=10)

# Semaphore to limit concurrent OCR operations to 5
OCR_CONCURRENCY_LIMIT = 5

# Retry configuration
OCR_MAX_RETRIES = 3


async def async_process_single_image_url(image_url: str, document_type: str = "unknown") -> dict:
    """
    Async wrapper for process_single_image_url.
    Downloads image and performs OCR in a thread pool to avoid blocking.
    Retries up to 3 times if page_no or survey_numbers are missing.
    
    Args:
        image_url: URL of the image to process
        document_type: Type of document for OCR prompt
    
    Returns:
        dict: OCR result with image_url, page_no, survey_numbers, or error
    """
    loop = asyncio.get_event_loop()
    
    def _sync_ocr_with_retry():
        """Synchronous OCR processing with retry logic."""
        logger.info(f"[Async OCR] Processing image URL: {image_url}")
        
        # Download image once (reuse for retries)
        try:
            response = requests.get(image_url, timeout=60)
            response.raise_for_status()
            image_bytes = response.content
        except Exception as e:
            logger.error(f"[Async OCR] Failed to download image {image_url}: {e}")
            return {"image_url": image_url, "error": str(e)}
        
        last_result = None
        
        for attempt in range(1, OCR_MAX_RETRIES + 1):
            try:
                image = Image.open(io.BytesIO(image_bytes))
                
                # Perform OCR
                ocr_result_text = gemini_ocr(image, 1, prompt_type=document_type)
                
                # Try to parse JSON
                try:
                    cleaned_text = ocr_result_text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(cleaned_text)
                except json.JSONDecodeError:
                    logger.warning(f"[Async OCR] Attempt {attempt}/{OCR_MAX_RETRIES}: Failed to parse JSON from Gemini response: {ocr_result_text}")
                    data = {}  # Empty dict - keys are missing

                result = {
                    "image_url": image_url,
                    "page_no": data.get("page_no"),
                    "survey_numbers": data.get("survey_numbers", [])
                }
                last_result = result
                
                # Check if keys exist in the response (values can be empty)
                has_page_no_key = "page_no" in data
                has_survey_numbers_key = "survey_numbers" in data
                
                if has_page_no_key and has_survey_numbers_key:
                    logger.info(f"[Async OCR] Success on attempt {attempt}/{OCR_MAX_RETRIES}: page_no={data.get('page_no')}, survey_numbers={data.get('survey_numbers')}")
                    return result
                else:
                    missing = []
                    if not has_page_no_key:
                        missing.append("page_no")
                    if not has_survey_numbers_key:
                        missing.append("survey_numbers")
                    
                    if attempt < OCR_MAX_RETRIES:
                        logger.warning(f"[Async OCR] Attempt {attempt}/{OCR_MAX_RETRIES}: Missing {missing} for {image_url}. Retrying...")
                    else:
                        logger.error(f"[Async OCR] Attempt {attempt}/{OCR_MAX_RETRIES}: Missing {missing} for {image_url}. No more retries.")

            except Exception as e:
                logger.error(f"[Async OCR] Attempt {attempt}/{OCR_MAX_RETRIES}: Error processing {image_url}: {e}")
                last_result = {"image_url": image_url, "error": str(e)}
                
                if attempt >= OCR_MAX_RETRIES:
                    return last_result
        
        # Return the last result after all retries exhausted
        return last_result or {"image_url": image_url, "error": "All retries exhausted"}
    
    # Run the sync OCR in thread pool to not block event loop
    return await loop.run_in_executor(_ocr_executor, _sync_ocr_with_retry)


async def process_ocr_concurrent(
    url_list: list[str],
    document_type: str,
    on_result_callback: Callable[[dict], Any],
    max_concurrent: int = OCR_CONCURRENCY_LIMIT
) -> tuple[bool, list[dict]]:
    """
    Process OCR for multiple URLs concurrently with immediate callback on each result.
    
    This function runs up to `max_concurrent` OCR operations in parallel.
    As soon as each OCR completes, the `on_result_callback` is called with the result,
    allowing immediate database insertion without waiting for all OCRs to complete.
    
    Args:
        url_list: List of image URLs to process
        document_type: Type of document for OCR prompt selection
        on_result_callback: Callback function called immediately when each OCR completes.
                           Signature: callback(ocr_result: dict) -> bool
                           Should return True to continue processing, False to stop (e.g., on error)
        max_concurrent: Maximum number of concurrent OCR operations (default: 5)
    
    Returns:
        tuple: (all_success: bool, results: list[dict])
        - all_success: True if all OCRs completed successfully and callbacks returned True
        - results: List of all OCR results (may be partial if stopped early)
    """
    logger.info(f"[Concurrent OCR] Starting OCR for {len(url_list)} URLs | Max Concurrent: {max_concurrent}")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    all_success = True
    stop_processing = False
    results_lock = asyncio.Lock()
    
    async def process_with_callback(url: str, index: int):
        nonlocal all_success, stop_processing
        
        # Check if we should stop (another task failed)
        if stop_processing:
            return None
        
        async with semaphore:
            # Double-check stop flag after acquiring semaphore
            if stop_processing:
                return None
            
            logger.info(f"[Concurrent OCR] Processing URL {index + 1}/{len(url_list)}: {url}")
            
            try:
                # Perform async OCR
                result = await async_process_single_image_url(url, document_type)
                
                # Immediately call the callback with the result
                async with results_lock:
                    if not stop_processing:
                        results.append(result)
                        
                        # Call callback - if it returns False, stop processing
                        should_continue = on_result_callback(result)
                        
                        if not should_continue:
                            logger.warning(f"[Concurrent OCR] Callback returned False for {url} - stopping further processing")
                            all_success = False
                            stop_processing = True
                        else:
                            logger.info(f"[Concurrent OCR] Result processed for {url}: page_no={result.get('page_no')}")
                
                return result
                
            except Exception as e:
                logger.error(f"[Concurrent OCR] Error processing {url}: {e}")
                error_result = {"image_url": url, "error": str(e)}
                
                async with results_lock:
                    results.append(error_result)
                    all_success = False
                    stop_processing = True
                
                return error_result
    
    # Create tasks for all URLs
    tasks = [process_with_callback(url, i) for i, url in enumerate(url_list)]
    
    # Run all tasks concurrently (semaphore limits actual parallelism to max_concurrent)
    await asyncio.gather(*tasks, return_exceptions=True)
    
    logger.info(f"[Concurrent OCR] Completed. Processed: {len(results)}/{len(url_list)}, Success: {all_success}")
    return all_success, results


def process_single_image_url(image_url, document_type="unknown"):
    """
    Downloads an image from a URL, performs OCR, and returns the results.
    """
    logger.info(f"Processing image URL: {image_url}")
    try:
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        
        # We pass page_no=1 as a placeholder since it's a single image
        ocr_result_text = gemini_ocr(image, 1, prompt_type=document_type)
        
        # Try to parse JSON
        try:
            # Clean possible markdown formatting
            cleaned_text = ocr_result_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from Gemini response: {ocr_result_text}")
            data = {"page_no": None, "survey_numbers": []}

        result = {
            "image_url": image_url,
            "page_no": data.get("page_no"),
            "survey_numbers": data.get("survey_numbers", [])
        }
        return result

    except Exception as e:
        logger.error(f"Error processing image URL {image_url}: {e}")
        return {"image_url": image_url, "error": str(e)}


def process_multiple_image_urls(url_list: list[str], document_type: str = "unknown", max_workers: int = 10):
    """
    Process multiple image URLs in parallel using ThreadPoolExecutor.
    
    Args:
        url_list: List of image URLs to process
        document_type: Type of document for OCR prompt
        max_workers: Number of parallel workers (default: 10)
    
    Returns:
        list: List of OCR results for each URL
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    logger.info(f"Processing {len(url_list)} URLs | Document Type: {document_type} | Workers: {max_workers}")
    
    results = []
    
    def process_single(url):
        """Process a single URL and return result."""
        try:
            result = process_single_image_url(url, document_type)
            return result
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return {"image_url": url, "error": str(e)}
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(process_single, url): url for url in url_list}
        
        # Collect results as they complete
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed: {url}")
            except Exception as e:
                logger.error(f"Future error for {url}: {e}")
                results.append({"image_url": url, "error": str(e)})
    
    logger.info(f"Completed processing {len(results)} URLs")
    return results

if __name__ == "__main__":
    # pdf_url = "http://182.18.157.124/BhargoUploadFiles/CCLA/DefaultUploads/NLR-B806-02-Nellore_Chintha%20Reddy%20palem_F-1416_20260107153017202.pdf"
    # images = split_pdf_to_images(pdf_url=pdf_url,save_dir="images")
    img_url = "http://182.18.157.124/BhargoUploadFiles/CCLA/1234/4/page_5_20260108144445647.png" 
    # print(f"Total images extracted: {len(images)}")
    
    # for i, img in enumerate(images):
    #     page_num = i + 1
    #     print(f"--- Processing Page {page_num} ---")
    #     # ocr_result = gemini_ocr(img, page_num, prompt_type="unknown")
    
    print(f"--- Processing Image URL ---")
    ocr_result = process_single_image_url(img_url,type="unknown")
    print(f"OCR Result:\n{ocr_result}\n")