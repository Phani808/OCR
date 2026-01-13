import pymssql, os
from dotenv import load_dotenv
import httpx
from PIL import Image
import fitz  # PyMuPDF
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from utils import process_single_image_url, process_ocr_concurrent

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

load_dotenv()

# Configuration
MAX_WORKERS = 10  # Number of parallel downloads
UPLOAD_API_URL = "http://182.18.157.124/CCLAAPI/V1/UploadImageVideoFiles/ASPXUpload"
USER_ID = "12345"  # Default UserID for upload

# Database table configuration
DB_NAME = os.getenv('DB_NAME', 'CCLA')  # Database name
DB_SCHEMA = os.getenv('DB_SCHEMA', 'dbo')  # Schema name

# Table names (using f-strings with DB_NAME and DB_SCHEMA)
TABLE_BOOK_UPLOAD = f"[{DB_NAME}].[{DB_SCHEMA}].[Book_Upload_Copy]"
TABLE_PAGE_UPLOAD = f"[{DB_NAME}].[{DB_SCHEMA}].[Page_Upload_Copy]"
TABLE_PAGE_UPLOAD_DOC_DATA = f"[{DB_NAME}].[{DB_SCHEMA}].[Page_Upload_document_data_Copy]"


# Fetch book upload data from database
def fetch_book_upload_data():
    conn = None
    try:
        conn = pymssql.connect(
            server=os.getenv('DB_SERVER'), port=os.getenv('DB_PORT'),
            user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE')
        )
        cursor = conn.cursor(as_dict=True)
        logger.info("Connected to database")
        cursor.execute(f"""
            SELECT
                [Bhargo_Emp_Id], [Bhargo_PostID], [Bhargo_Trans_Id],
                [district], [district_id],
                [mandal], [mandal_id], [revenuevillage], [revenuevillage_id],
                [document_type], [document_typeid], [fasil_year], [fasil_yearid],
                [voulme_no], [voulme_noid], [document_upload],
                [Bhargo_Is_Active], [Bhargo_GPS], [Bhargo_IME], [Bhargo_DId],
                [Bhargo_Employee_Location], [TransactionDate], [TransactionMonth], [TransactionYear],
                [book_status], [book_statusid]
            FROM {TABLE_BOOK_UPLOAD}
            WHERE [book_status] = 'Pending'
        """)
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"DB fetch failed: {e}")
        return []
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")


def update_book_status(trans_id: int, status: str, status_id: int):
    """
    Update the book_status and book_statusid in Book_Upload table.
    
    Args:
        trans_id: The Bhargo_Trans_Id of the record to update
        status: New status string ('Completed', 'Page No Not Found', etc.)
        status_id: New status ID (2=Completed, 3=Page No Not Found)
    """
    conn = None
    try:
        conn = pymssql.connect(
            server=os.getenv('DB_SERVER'), port=os.getenv('DB_PORT'),
            user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE')
        )
        cursor = conn.cursor()
        
        sql = f"""
            UPDATE {TABLE_BOOK_UPLOAD}
            SET [book_status] = %s, [book_statusid] = %s
            WHERE [Bhargo_Trans_Id] = %s
        """
        
        cursor.execute(sql, (status, status_id, trans_id))
        conn.commit()
        logger.info(f"Updated Book_Upload status: Trans_Id={trans_id}, status={status}, status_id={status_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update book status for Trans_Id={trans_id}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def insert_page_record(record: dict, ocr_result: dict):
    """
    Insert a page record into Page_Upload_Copy table.
    
    Args:
        record: Original record from Book_Upload with metadata
        ocr_result: OCR result with image_url, page_no, survey_numbers
    """
    conn = None
    try:
        conn = pymssql.connect(
            server=os.getenv('DB_SERVER'), port=os.getenv('DB_PORT'),
            user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE')
        )
        cursor = conn.cursor()
        
        # Extract values from OCR result
        page_no = ocr_result.get('page_no')
        image_url = ocr_result.get('image_url')
        
        # Extract metadata from original record
        sql = f"""
            INSERT INTO {TABLE_PAGE_UPLOAD} (
                [Bhargo_Emp_Id], [Bhargo_PostID], [Bhargo_Trans_Date],
                [district], [district_id],
                [mandal], [mandal_id],
                [revenuevillage], [revenuevillage_id],
                [document_type], [document_typeid],
                [fasil_year], [fasil_yearid],
                [voulme_no], [voulme_noid],
                [page_no], [document_upload],
                [page_status], [page_statusid],
                [Bhargo_Is_Active], [Bhargo_GPS], [Bhargo_IME], [Bhargo_DId],
                [Bhargo_Employee_Location], [TransactionDate], [TransactionMonth], [TransactionYear]
            ) VALUES (
                %s, %s, GETDATE(),
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                'Pending at VRO', '1',
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
        """
        
        values = (
            record.get('Bhargo_Emp_Id'),
            record.get('Bhargo_PostID'),
            record.get('district'),
            record.get('district_id'),
            record.get('mandal'),
            record.get('mandal_id'),
            record.get('revenuevillage'),
            record.get('revenuevillage_id'),
            record.get('document_type'),
            record.get('document_typeid'),
            record.get('fasil_year'),
            record.get('fasil_yearid'),
            record.get('voulme_no'),
            record.get('voulme_noid'),
            page_no,
            image_url,
            record.get('Bhargo_Is_Active'),
            record.get('Bhargo_GPS'),
            record.get('Bhargo_IME'),
            record.get('Bhargo_DId'),
            record.get('Bhargo_Employee_Location'),
            record.get('TransactionDate'),
            record.get('TransactionMonth'),
            record.get('TransactionYear')
        )
        
        cursor.execute(sql, values)
        
        # Get the new Bhargo_Trans_Id (auto-generated)
        cursor.execute("SELECT SCOPE_IDENTITY()")
        new_trans_id = cursor.fetchone()[0]
        
        conn.commit()
        logger.info(f"Inserted page record: page_no={page_no}, image_url={image_url}, new_trans_id={new_trans_id}")
        return new_trans_id  # Return the new Trans_Id for survey number insert
        
    except Exception as e:
        logger.error(f"Failed to insert page record: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def insert_survey_numbers(record: dict, new_page_trans_id: int, survey_numbers: list):
    """
    Insert survey numbers into Page_Upload_document_data_Copy table.
    Each survey number is inserted as a separate row with all metadata.
    
    Args:
        record: Original record from Book_Upload with metadata
        new_page_trans_id: The Trans_Id from newly inserted Page_Upload_Copy
        survey_numbers: List of survey numbers from OCR result
    """
    if not survey_numbers:
        logger.info("No survey numbers to insert")
        return True
    
    conn = None
    try:
        conn = pymssql.connect(
            server=os.getenv('DB_SERVER'), port=os.getenv('DB_PORT'),
            user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE')
        )
        cursor = conn.cursor()
        
        sql = f"""
            INSERT INTO {TABLE_PAGE_UPLOAD_DOC_DATA} (
                [Bhargo_Emp_Id], [Bhargo_PostID], [Bhargo_Trans_Date],
                [Bhargo_Ref_TransID], [document_data_survery_no],
                [Bhargo_Is_Active], [Bhargo_GPS], [Bhargo_IME], [Bhargo_DId],
                [Bhargo_Employee_Location], [TransactionDate], [TransactionMonth], [TransactionYear]
            ) VALUES (
                %s, %s, GETDATE(),
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s
            )
        """
        
        inserted_count = 0
        for survey_no in survey_numbers:
            if survey_no and str(survey_no).strip():
                values = (
                    record.get('Bhargo_Emp_Id'),
                    record.get('Bhargo_PostID'),
                    new_page_trans_id,
                    str(survey_no).strip(),
                    record.get('Bhargo_Is_Active'),
                    record.get('Bhargo_GPS'),
                    record.get('Bhargo_IME'),
                    record.get('Bhargo_DId'),
                    record.get('Bhargo_Employee_Location'),
                    record.get('TransactionDate'),
                    record.get('TransactionMonth'),
                    record.get('TransactionYear')
                )
                cursor.execute(sql, values)
                inserted_count += 1
        
        conn.commit()
        logger.info(f"Inserted {inserted_count} survey numbers for Ref_TransID={new_page_trans_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to insert survey numbers: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


async def download_pdf(client: httpx.AsyncClient, pdf_url: str) -> bytes | None:
    """Download PDF content using httpx async client."""
    try:
        response = await client.get(pdf_url, timeout=60.0)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download PDF from {pdf_url}: {e}")
        return None


def convert_pdf_to_image_bytes(pdf_content: bytes) -> list[tuple[str, bytes]]:
    """Convert PDF bytes to list of (filename, image_bytes) tuples."""
    try:
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        image_data_list = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Convert image to bytes (PNG format for API compatibility)
            img_buffer = BytesIO()
            img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            
            filename = f"page_{page_num+1}.png"
            image_data_list.append((filename, img_bytes))
        
        pdf_document.close()
        return image_data_list
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        return []


async def upload_image(client: httpx.AsyncClient, filename: str, image_bytes: bytes, folder_name: str) -> str | None:
    """Upload a single image to the API and return the ServerPath."""
    try:
        files = {"Files": (filename, image_bytes, "image/png")}
        data = {"UserID": USER_ID, "FolderName": folder_name}
        
        response = await client.post(UPLOAD_API_URL, data=data, files=files, timeout=60.0)
        response.raise_for_status()
        
        result = response.json()
        if result.get("Status") == "200" and result.get("FileSavedPaths"):
            server_path = result["FileSavedPaths"][0].get("ServerPath")
            logger.info(f"Uploaded {filename} -> {server_path}")
            return server_path
        else:
            logger.error(f"Upload failed for {filename}: {result}")
            return None
    except Exception as e:
        logger.error(f"Failed to upload {filename}: {e}")
        return None


async def upload_images_batch(client: httpx.AsyncClient, image_data_list: list[tuple[str, bytes]], folder_name: str) -> list[str]:
    """Upload multiple images in parallel and return list of ServerPaths."""
    tasks = [upload_image(client, filename, img_bytes, folder_name) for filename, img_bytes in image_data_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out errors and None values
    server_paths = [r for r in results if isinstance(r, str) and r]
    return server_paths


async def process_record(client: httpx.AsyncClient, record: dict, executor: ThreadPoolExecutor) -> list[str]:
    """Process a single record: download PDF, convert to images, upload to API, return ServerPaths."""
    pdf_url = record.get('document_upload')
    if not pdf_url:
        logger.warning(f"No document_upload URL for record: {record.get('Bhargo_Emp_Id')}")
        return []
    
    emp_id = record.get('Bhargo_Emp_Id')
    trans_id = record.get('Bhargo_Trans_Id', 'unknown')
    district = record.get('district')
    logger.info(f"Processing document for Emp_Id: {emp_id}, Trans_Id: {trans_id}, District: {district}")
    
    # Download PDF
    pdf_content = await download_pdf(client, pdf_url)
    if not pdf_content:
        return []
    
    logger.info(f"Downloaded PDF for Emp_Id: {emp_id}")
    
    # Convert PDF to images in thread pool (CPU-bound operation)
    loop = asyncio.get_event_loop()
    image_data_list = await loop.run_in_executor(executor, convert_pdf_to_image_bytes, pdf_content)
    
    if not image_data_list:
        logger.error(f"No images extracted for Emp_Id: {emp_id}")
        return []
    
    logger.info(f"Extracted {len(image_data_list)} pages for Emp_Id: {emp_id}")
    
    # Upload images to API
    folder_name = str(trans_id)
    server_paths = await upload_images_batch(client, image_data_list, folder_name)
    
    logger.info(f"Uploaded {len(server_paths)} images for Emp_Id: {emp_id}")
    return server_paths


async def process_all_records(records: list) -> dict[str, list[str]]:
    """Process all records in parallel with controlled concurrency.
    Returns dict mapping Trans_Id to list of ServerPaths.
    """
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    results_map = {}
    
    async def limited_process(client, record, executor):
        async with semaphore:
            trans_id = str(record.get('Bhargo_Trans_Id', 'unknown'))
            document_type = record.get('document_type', 'unknown')
            paths = await process_record(client, record, executor)
            return trans_id, document_type, paths
    
    async with httpx.AsyncClient() as client:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [limited_process(client, record, executor) for record in records]
            results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Build results map with document_type
    total_uploads = 0
    errors = 0
    for result in results:
        if isinstance(result, Exception):
            errors += 1
            logger.error(f"Record processing error: {result}")
        elif isinstance(result, tuple):
            trans_id, document_type, paths = result
            results_map[trans_id] = {
                "document_type": document_type,
                "urls": paths
            }
            total_uploads += len(paths)
    
    logger.info(f"Processing complete. Total images uploaded: {total_uploads}, Errors: {errors}")
    return results_map


def get_url_list(results_map: dict) -> list[dict]:
    """Convert results dict to a list with URLs and document_type."""
    url_list = []
    for trans_id, data in results_map.items():
        document_type = data.get("document_type", "unknown")
        for url in data.get("urls", []):
            url_list.append({
                "url": url,
                "document_type": document_type,
                "trans_id": trans_id
            })
    return url_list


def process_url_list_for_record(url_list: list[str], document_type: str, record: dict):
    """
    Process URLs sequentially for a single record.
    Checks page_no immediately after each OCR - if missing, stops and marks as failed.
    Only inserts into database if ALL pages have valid page numbers.
    
    Returns:
        tuple: (success: bool, results: list)
        - success: True if all pages have page numbers and were inserted
        - results: List of OCR results (may be partial if stopped early)
    """
    trans_id = record.get('Bhargo_Trans_Id')
    logger.info(f"Processing {len(url_list)} URLs for Trans_Id={trans_id} | Document Type: {document_type}")
    
    results = []
    page_no_missing = False
    failed_url = None
    
    # Step 1: OCR each page and check page_no AND survey_numbers immediately - STOP if either is missing
    for url in url_list:
        logger.info(f"Trans_Id={trans_id}: Processing page {len(results) + 1}/{len(url_list)}: {url}")
        
        try:
            result = process_single_image_url(url, document_type)
            
            # Immediately check if page_no exists
            page_no = result.get('page_no')
            if page_no is None or str(page_no).strip() == '':
                logger.error(f"Trans_Id={trans_id}: Page number NOT FOUND for {url} - STOPPING processing")
                page_no_missing = True
                failed_url = url
                break  # Stop processing this PDF immediately
            
            # Also check if survey_numbers exist
            survey_numbers = result.get('survey_numbers', [])
            if not survey_numbers or len(survey_numbers) == 0:
                logger.error(f"Trans_Id={trans_id}: Survey numbers NOT FOUND for {url} - STOPPING processing")
                page_no_missing = True
                failed_url = url
                break  # Stop processing this PDF immediately
            
            logger.info(f"Trans_Id={trans_id}: Page {page_no} with {len(survey_numbers)} survey numbers found for {url}")
            results.append(result)
            
        except Exception as e:
            logger.error(f"Trans_Id={trans_id}: Error processing {url}: {e} - STOPPING processing")
            page_no_missing = True
            failed_url = url
            break  # Stop processing on any error
    
    # Step 2: If any page is missing page_no or survey_numbers, mark as failed and exit
    if page_no_missing:
        logger.warning(f"Trans_Id={trans_id}: Marking as 'Page No Or Survey No Not Found' due to missing data at {failed_url}")
        update_book_status(trans_id, "Page No Or Survey No Not Found", 3)
        return False, results
    
    # Step 3: All pages have page numbers - insert into database
    logger.info(f"Trans_Id={trans_id}: All {len(results)} pages have valid page numbers. Inserting into database...")
    
    insert_success = True
    for result in results:
        # Insert page record
        new_trans_id = insert_page_record(record, result)
        
        if new_trans_id:
            logger.info(f"Inserted: page_no={result.get('page_no')}, url={result.get('image_url')}, new_trans_id={new_trans_id}")
            
            # Insert survey numbers linked to the new page record
            survey_numbers = result.get('survey_numbers', [])
            if survey_numbers:
                insert_survey_numbers(record, new_trans_id, survey_numbers)
        else:
            logger.error(f"Failed to insert page record for {result.get('image_url')}")
            insert_success = False
    
    if insert_success:
        # All pages inserted successfully - mark as "Completed"
        update_book_status(trans_id, "Completed", 2)
        logger.info(f"Trans_Id={trans_id}: Processing completed successfully!")
    else:
        logger.error(f"Trans_Id={trans_id}: Some inserts failed, status not updated to Completed")
    
    return insert_success, results


async def async_process_url_list_for_record(url_list: list[str], document_type: str, record: dict):
    """
    Process URLs concurrently (5 at a time) for a single record with immediate DB insertion.
    
    This function:
    1. Runs up to 5 OCR operations in parallel
    2. As soon as each OCR completes, immediately inserts the result into the database
    3. If any result is missing page_no or survey_numbers, stops all further processing
    4. Updates final book status based on overall success
    
    Args:
        url_list: List of image URLs to process
        document_type: Type of document for OCR prompt
        record: Original record from Book_Upload with metadata
    
    Returns:
        tuple: (success: bool, results: list)
        - success: True if all pages processed and inserted successfully
        - results: List of OCR results (may be partial if stopped early)
    """
    trans_id = record.get('Bhargo_Trans_Id')
    logger.info(f"[Async] Processing {len(url_list)} URLs for Trans_Id={trans_id} | Document Type: {document_type} | Max Concurrent: 5")
    
    inserted_results = []
    failed = False
    failed_url = None
    
    def on_ocr_result(result: dict) -> bool:
        """
        Callback called immediately when each OCR completes.
        Validates the result and inserts into database right away.
        
        Returns:
            bool: True to continue processing other URLs, False to stop processing
        """
        nonlocal failed, failed_url
        
        image_url = result.get('image_url')
        
        # Check for errors
        if 'error' in result:
            logger.error(f"Trans_Id={trans_id}: OCR error for {image_url}: {result.get('error')}")
            failed = True
            failed_url = image_url
            return False  # Stop processing
        
        # Validate page_no
        page_no = result.get('page_no')
        if page_no is None or str(page_no).strip() == '':
            logger.error(f"Trans_Id={trans_id}: Page number NOT FOUND for {image_url}")
            failed = True
            failed_url = image_url
            return False  # Stop processing
        
        # Validate survey_numbers
        survey_numbers = result.get('survey_numbers', [])
        if not survey_numbers or len(survey_numbers) == 0:
            logger.error(f"Trans_Id={trans_id}: Survey numbers NOT FOUND for {image_url}")
            failed = True
            failed_url = image_url
            return False  # Stop processing
        
        # All validations passed - INSERT IMMEDIATELY into database
        logger.info(f"Trans_Id={trans_id}: Inserting page {page_no} with {len(survey_numbers)} survey numbers immediately")
        
        new_trans_id = insert_page_record(record, result)
        
        if new_trans_id:
            logger.info(f"Trans_Id={trans_id}: Inserted page_no={page_no}, url={image_url}, new_trans_id={new_trans_id}")
            
            # Insert survey numbers linked to the new page record
            insert_survey_numbers(record, new_trans_id, survey_numbers)
            inserted_results.append(result)
            return True  # Continue processing
        else:
            logger.error(f"Trans_Id={trans_id}: Failed to insert page record for {image_url}")
            failed = True
            failed_url = image_url
            return False  # Stop processing
    
    # Run concurrent OCR with immediate callback
    all_success, results = await process_ocr_concurrent(
        url_list=url_list,
        document_type=document_type,
        on_result_callback=on_ocr_result,
        max_concurrent=5
    )
    
    # Update final book status
    if failed:
        logger.warning(f"Trans_Id={trans_id}: Marking as 'Page No Or Survey No Not Found' due to error at {failed_url}")
        update_book_status(trans_id, "Page No Or Survey No Not Found", 3)
        return False, inserted_results
    
    if all_success and len(inserted_results) == len(url_list):
        update_book_status(trans_id, "Completed", 2)
        logger.info(f"Trans_Id={trans_id}: Processing completed successfully! All {len(inserted_results)} pages inserted.")
        return True, inserted_results
    else:
        logger.error(f"Trans_Id={trans_id}: Some operations failed. Inserted: {len(inserted_results)}/{len(url_list)}")
        return False, inserted_results


if __name__ == "__main__":
    # Fetch all pending book records from database
    records = fetch_book_upload_data()
    logger.info(f"Fetched {len(records)} pending records from database")
    
    if records:
        # Process each record individually
        for record in records:
            trans_id = record.get('Bhargo_Trans_Id')
            document_type = record.get('document_type', 'unknown')
            
            logger.info(f"=" * 60)
            logger.info(f"Processing record: Trans_Id={trans_id}, Document Type={document_type}")
            
            # Run async processing to upload PDF and get image URLs for this record
            results = asyncio.run(process_all_records([record]))
            
            # Get URL list for this record
            url_list = []
            for tid, data in results.items():
                url_list.extend(data.get("urls", []))
            
            if url_list:
                logger.info(f"Trans_Id={trans_id}: Extracted {len(url_list)} page images")
                
                # Process OCR with concurrent processing (5 at a time) and immediate DB insertion
                success, ocr_results = asyncio.run(
                    async_process_url_list_for_record(url_list, document_type, record)
                )
                
                if success:
                    logger.info(f"Trans_Id={trans_id}: Successfully processed and marked as Completed")
                else:
                    logger.warning(f"Trans_Id={trans_id}: Processing failed or pages missing page numbers")
            else:
                logger.error(f"Trans_Id={trans_id}: No pages extracted from PDF, marking as Page No Not Found")
                update_book_status(trans_id, "Page No Not Found", 3)
            
            logger.info(f"=" * 60)
    else:
        logger.warning("No pending records found to process")