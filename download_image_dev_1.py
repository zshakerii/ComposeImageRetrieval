import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


# ============================================================
# CONFIGURATION
# ============================================================

JSON_FILE = (
    r"C:\Users\user\Desktop\Python\ImageRetrieval"
    r"\nlvr\nlvr2\data\dev.json"
)

SAVE_DIR = (
    r"C:\Users\user\Desktop\Python\ImageRetrieval"
    r"\nlvr\nlvr2\images\dev"
)

FAILED_FILE = os.path.join(
    SAVE_DIR,
    "failed_downloads.json"
)

LOG_FILE = os.path.join(
    SAVE_DIR,
    "download_log.json"
)

# تعداد دانلودهای همزمان
MAX_WORKERS = 4

# تعداد تلاش برای هر URL
MAX_RETRIES = 5

# timeout:
# connect timeout = 15 sec
# read timeout = 45 sec
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 45

# فاصله اولیه retry
INITIAL_BACKOFF = 2

# حداکثر size برای chunk
CHUNK_SIZE = 64 * 1024


# ============================================================
# USER AGENTS
# ============================================================

USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
]


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# THREAD LOCAL SESSION
# ============================================================

thread_local = threading.local()


def get_session():

    if not hasattr(thread_local, "session"):

        session = requests.Session()

        retry_strategy = Retry(
            total=0,
            connect=0,
            read=0,
            redirect=0,
            status=0
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )

        session.mount(
            "http://",
            adapter
        )

        session.mount(
            "https://",
            adapter
        )

        thread_local.session = session

    return thread_local.session


# ============================================================
# LOAD JSON LINES
# ============================================================

def load_examples():

    examples = []

    with open(
        JSON_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                example = json.loads(line)

            except json.JSONDecodeError as ex:

                print(
                    f"[ERROR] Invalid JSON "
                    f"at line {line_number}: {ex}"
                )

                continue

            examples.append(example)

    return examples


# ============================================================
# BUILD IMAGE LIST
# ============================================================

def build_image_list(examples):

    images = {}

    invalid_count = 0

    for example in examples:

        identifier = example.get(
            "identifier"
        )

        if not identifier:
            invalid_count += 1
            continue

        parts = identifier.split("-")

        # Example:
        #
        # dev-149-2-0
        #
        # => dev-149-2
        #
        if len(parts) < 3:

            print(
                f"[WARNING] Invalid identifier: "
                f"{identifier}"
            )

            invalid_count += 1

            continue

        image_id = "-".join(
            parts[:3]
        )

        left_url = example.get(
            "left_url"
        )

        right_url = example.get(
            "right_url"
        )

        # ----------------------------------------------------
        # LEFT IMAGE
        # ----------------------------------------------------

        if (
            left_url
            and isinstance(left_url, str)
            and left_url.startswith(
                ("http://", "https://")
            )
        ):

            filename = (
                f"{image_id}-img0.png"
            )

            # اگر قبلاً در JSON دیگری همان filename آمده
            # و URL متفاوت باشد، اولین URL را نگه می‌داریم.
            if filename not in images:

                images[filename] = {
                    "url": left_url,
                    "identifier": identifier,
                    "side": "left"
                }

        # ----------------------------------------------------
        # RIGHT IMAGE
        # ----------------------------------------------------

        if (
            right_url
            and isinstance(right_url, str)
            and right_url.startswith(
                ("http://", "https://")
            )
        ):

            filename = (
                f"{image_id}-img1.png"
            )

            if filename not in images:

                images[filename] = {
                    "url": right_url,
                    "identifier": identifier,
                    "side": "right"
                }

    if invalid_count > 0:

        print(
            f"[WARNING] Invalid records: "
            f"{invalid_count}"
        )

    return images


# ============================================================
# CHECK EXISTING IMAGE
# ============================================================

def is_existing_file(path):

    return (
        os.path.isfile(path)
        and os.path.getsize(path) > 0
    )


# ============================================================
# VALIDATE IMAGE
# ============================================================

def validate_image_file(path):

    try:

        with Image.open(path) as image:

            # verify() ساختار فایل را بررسی می‌کند
            image.verify()

        # دوباره باز می‌کنیم تا format و اندازه را نیز
        # بررسی کنیم.
        with Image.open(path) as image:

            width, height = image.size
            image_format = image.format

            if width <= 0 or height <= 0:

                return {
                    "valid": False,
                    "error": "Invalid image dimensions",
                    "format": image_format,
                    "width": width,
                    "height": height
                }

            return {
                "valid": True,
                "error": None,
                "format": image_format,
                "width": width,
                "height": height
            }

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as ex:

        return {
            "valid": False,
            "error": str(ex),
            "format": None,
            "width": None,
            "height": None
        }


# ============================================================
# CHECK EXISTING FILE
# ============================================================

def check_existing_image(path):

    # --------------------------------------------------------
    # فایل وجود ندارد
    # --------------------------------------------------------

    if not os.path.isfile(path):

        return {
            "exists": False,
            "valid": False
        }

    # --------------------------------------------------------
    # فایل zero-byte
    # --------------------------------------------------------

    if os.path.getsize(path) == 0:

        return {
            "exists": True,
            "valid": False,
            "reason": "zero_byte"
        }

    # --------------------------------------------------------
    # بررسی تصویر
    # --------------------------------------------------------

    validation = validate_image_file(
        path
    )

    return {
        "exists": True,
        "valid": validation["valid"],
        "reason": validation.get(
            "error"
        ),
        "format": validation.get(
            "format"
        ),
        "width": validation.get(
            "width"
        ),
        "height": validation.get(
            "height"
        )
    }


# ============================================================
# GET RESPONSE CONTENT
# ============================================================

def download_to_temp(
    session,
    url,
    temp_path
):

    user_agent = random.choice(
        USER_AGENTS
    )

    headers = {
        "User-Agent": user_agent,
        "Accept": (
            "image/avif,image/webp,image/apng,"
            "image/svg+xml,image/*,*/*;q=0.8"
        ),
        "Accept-Language": (
            "en-US,en;q=0.9"
        ),
        "Connection": "keep-alive",
        "Referer": url
    }

    response = session.get(
        url,
        headers=headers,
        timeout=(
            CONNECT_TIMEOUT,
            READ_TIMEOUT
        ),
        stream=True,
        allow_redirects=True
    )

    status_code = response.status_code

    content_type = (
        response.headers.get(
            "Content-Type",
            ""
        )
        .lower()
        .split(";")[0]
        .strip()
    )

    content_length = response.headers.get(
        "Content-Length"
    )

    # --------------------------------------------------------
    # HTTP status
    # --------------------------------------------------------

    if status_code != 200:

        response.close()

        raise DownloadError(
            f"HTTP {status_code}",
            status_code=status_code,
            content_type=content_type
        )

    # --------------------------------------------------------
    # Content-Type
    #
    # بعض سرورها Content-Type مناسب نمی‌دهند.
    # در صورتی که خالی باشد آن را رد نمی‌کنیم؛
    # ولی اگر HTML باشد، احتمالاً صفحه خطاست.
    # --------------------------------------------------------

    if content_type:

        if (
            "text/html" in content_type
            or "application/json" in content_type
            or "text/plain" in content_type
        ):

            response.close()

            raise DownloadError(
                f"Unexpected Content-Type: "
                f"{content_type}",
                status_code=status_code,
                content_type=content_type
            )

    # --------------------------------------------------------
    # Save file
    # --------------------------------------------------------

    total_bytes = 0

    try:

        with open(
            temp_path,
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=CHUNK_SIZE
            ):

                if not chunk:
                    continue

                file.write(chunk)

                total_bytes += len(chunk)

    finally:

        response.close()

    # --------------------------------------------------------
    # Empty response
    # --------------------------------------------------------

    if total_bytes == 0:

        raise DownloadError(
            "Empty response",
            status_code=status_code,
            content_type=content_type
        )

    return {
        "status_code": status_code,
        "content_type": content_type,
        "content_length": content_length,
        "downloaded_bytes": total_bytes
    }


# ============================================================
# CUSTOM DOWNLOAD ERROR
# ============================================================

class DownloadError(Exception):

    def __init__(
        self,
        message,
        status_code=None,
        content_type=None
    ):

        super().__init__(message)

        self.status_code = status_code
        self.content_type = content_type


# ============================================================
# DOWNLOAD ONE IMAGE
# ============================================================

def download_image(
    filename,
    image_info
):

    url = image_info["url"]
    identifier = image_info.get(
        "identifier"
    )
    side = image_info.get(
        "side"
    )

    final_path = os.path.join(
        SAVE_DIR,
        filename
    )

    temp_path = (
        final_path
        + ".part"
    )

    # ========================================================
    # FIRST CHECK
    #
    # اگر فایل موجود و سالم است:
    # اصلاً HTTP REQUEST نمی‌زنیم.
    # ========================================================

    existing = check_existing_image(
        final_path
    )

    if (
        existing["exists"]
        and existing["valid"]
    ):

        return {
            "status": "exists",
            "filename": filename,
            "url": url,
            "identifier": identifier,
            "side": side,
            "message": "File already exists"
        }

    # اگر فایل خراب/ناقص است، قبل از دانلود حذف می‌شود.
    if existing["exists"]:

        try:

            os.remove(final_path)

        except OSError:

            pass

    session = get_session()

    last_error = None
    last_status = None
    last_content_type = None

    # ========================================================
    # RETRY LOOP
    # ========================================================

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            # ------------------------------------------------
            # Remove old .part
            # ------------------------------------------------

            if os.path.exists(temp_path):

                try:

                    os.remove(temp_path)

                except OSError:

                    pass

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            result = download_to_temp(
                session=session,
                url=url,
                temp_path=temp_path
            )

            last_status = result.get(
                "status_code"
            )

            last_content_type = result.get(
                "content_type"
            )

            # ------------------------------------------------
            # Validate downloaded image
            # ------------------------------------------------

            validation = validate_image_file(
                temp_path
            )

            if not validation["valid"]:

                raise DownloadError(
                    "Downloaded file is not a valid image: "
                    f"{validation.get('error')}",
                    status_code=last_status,
                    content_type=last_content_type
                )

            # ------------------------------------------------
            # Atomic rename
            # ------------------------------------------------

            os.replace(
                temp_path,
                final_path
            )

            return {
                "status": "downloaded",
                "filename": filename,
                "url": url,
                "identifier": identifier,
                "side": side,
                "attempts": attempt,
                "http_status": last_status,
                "content_type": last_content_type,
                "image_format": validation.get(
                    "format"
                ),
                "width": validation.get(
                    "width"
                ),
                "height": validation.get(
                    "height"
                )
            }

        except DownloadError as ex:

            last_error = str(ex)
            last_status = ex.status_code
            last_content_type = ex.content_type

        except requests.exceptions.Timeout as ex:

            last_error = (
                f"Timeout: {str(ex)}"
            )

        except requests.exceptions.ConnectionError as ex:

            last_error = (
                f"ConnectionError: {str(ex)}"
            )

        except requests.exceptions.TooManyRedirects as ex:

            last_error = (
                f"TooManyRedirects: {str(ex)}"
            )

        except requests.exceptions.RequestException as ex:

            last_error = (
                f"RequestException: {str(ex)}"
            )

        except Exception as ex:

            last_error = (
                f"{type(ex).__name__}: "
                f"{str(ex)}"
            )

        # ----------------------------------------------------
        # Remove failed .part file
        # ----------------------------------------------------

        if os.path.exists(temp_path):

            try:

                os.remove(temp_path)

            except OSError:

                pass

        # ----------------------------------------------------
        # Decide retry
        # ----------------------------------------------------

        retryable = True

        # برای این statusها تلاش مجدد معمولاً
        # فایده‌ای ندارد.
        if last_status in {
            400,
            401,
            403,
            404,
            410
        }:

            retryable = False

        # 429, 500, 502, 503, 504
        # قابل retry هستند.
        if (
            retryable
            and attempt < MAX_RETRIES
        ):

            backoff = (
                INITIAL_BACKOFF
                * (2 ** (attempt - 1))
            )

            # jitter
            backoff += random.uniform(
                0,
                1
            )

            time.sleep(
                backoff
            )

        else:

            break

    # ========================================================
    # FAILED
    # ========================================================

    return {
        "status": "failed",
        "filename": filename,
        "url": url,
        "identifier": identifier,
        "side": side,
        "attempts": MAX_RETRIES,
        "http_status": last_status,
        "content_type": last_content_type,
        "error": last_error
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("NLVR2 DEV IMAGE DOWNLOADER")
    print("=" * 80)

    print(
        f"JSON file : {JSON_FILE}"
    )

    print(
        f"Save dir  : {SAVE_DIR}"
    )

    print(
        f"Workers   : {MAX_WORKERS}"
    )

    print(
        f"Retries   : {MAX_RETRIES}"
    )

    print("=" * 80)

    # ========================================================
    # CHECK JSON
    # ========================================================

    if not os.path.isfile(JSON_FILE):

        print(
            "[ERROR] JSON file not found:"
        )

        print(JSON_FILE)

        return

    # ========================================================
    # LOAD JSON
    # ========================================================

    print()
    print("Loading dev.json...")

    examples = load_examples()

    print(
        f"Examples loaded: {len(examples)}"
    )

    # ========================================================
    # BUILD IMAGE LIST
    # ========================================================

    print()
    print("Building image list...")

    images = build_image_list(
        examples
    )

    print(
        f"Unique images: {len(images)}"
    )

    # ========================================================
    # FIND EXISTING / MISSING
    # ========================================================

    existing_count = 0
    invalid_existing_count = 0

    images_to_download = {}

    print()
    print(
        "Checking existing files..."
    )

    for filename, info in images.items():

        path = os.path.join(
            SAVE_DIR,
            filename
        )

        existing = check_existing_image(
            path
        )

        # ----------------------------------------------------
        # موجود و سالم
        # ----------------------------------------------------

        if (
            existing["exists"]
            and existing["valid"]
        ):

            existing_count += 1

            continue

        # ----------------------------------------------------
        # موجود ولی خراب
        # ----------------------------------------------------

        if existing["exists"]:

            invalid_existing_count += 1

        # ----------------------------------------------------
        # نیاز به دانلود
        # ----------------------------------------------------

        images_to_download[
            filename
        ] = info

    print()
    print("-" * 80)

    print(
        f"Total images        : {len(images)}"
    )

    print(
        f"Existing & valid    : {existing_count}"
    )

    print(
        f"Existing but invalid: {invalid_existing_count}"
    )

    print(
        f"Need download       : {len(images_to_download)}"
    )

    print("-" * 80)

    # ========================================================
    # NOTHING TO DOWNLOAD
    # ========================================================

    if not images_to_download:

        print()
        print(
            "All required images already exist "
            "and are valid."
        )

        print(
            "No HTTP request was required."
        )

        return

    # ========================================================
    # DOWNLOAD
    # ========================================================

    results = []

    print()
    print(
        "Downloading missing images..."
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                download_image,
                filename,
                info
            ): filename

            for filename, info
            in images_to_download.items()
        }

        with tqdm(
            total=len(futures),
            desc="Downloading",
            unit="img"
        ) as progress:

            for future in as_completed(
                futures
            ):

                filename = futures[
                    future
                ]

                try:

                    result = future.result()

                except Exception as ex:

                    result = {
                        "status": "failed",
                        "filename": filename,
                        "error": (
                            f"{type(ex).__name__}: "
                            f"{str(ex)}"
                        )
                    }

                results.append(result)

                progress.update(1)

    # ========================================================
    # STATISTICS
    # ========================================================

    downloaded = [
        x
        for x in results
        if x["status"] == "downloaded"
    ]

    exists = [
        x
        for x in results
        if x["status"] == "exists"
    ]

    failed = [
        x
        for x in results
        if x["status"] == "failed"
    ]

    # ========================================================
    # SAVE FAILED RESULTS
    # ========================================================

    with open(
        FAILED_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            failed,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SAVE FULL LOG
    # ========================================================

    with open(
        LOG_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # PRINT FAILURE DETAILS
    # ========================================================

    print()
    print("=" * 80)
    print("DOWNLOAD FINISHED")
    print("=" * 80)

    print(
        f"Total images        : {len(images)}"
    )

    print(
        f"Already valid       : {existing_count}"
    )

    print(
        f"Invalid existing    : {invalid_existing_count}"
    )

    print(
        f"Downloaded now      : {len(downloaded)}"
    )

    print(
        f"Failed              : {len(failed)}"
    )

    # ========================================================
    # FAILURE BREAKDOWN
    # ========================================================

    if failed:

        print()
        print(
            "Failure details:"
        )

        status_counter = {}

        for item in failed:

            status = item.get(
                "http_status"
            )

            if status is None:

                key = (
                    item.get("error")
                    or "Unknown error"
                )

            else:

                key = f"HTTP {status}"

            status_counter[key] = (
                status_counter.get(
                    key,
                    0
                )
                + 1
            )

        for reason, count in sorted(
            status_counter.items(),
            key=lambda x: -x[1]
        ):

            print(
                f"  {reason}: {count}"
            )

    # ========================================================
    # FAILED FILE PATH
    # ========================================================

    if failed:

        print()
        print(
            "Failed downloads saved to:"
        )

        print(
            FAILED_FILE
        )

    print()
    print(
        "Full log saved to:"
    )

    print(
        LOG_FILE
    )

    print()
    print(
        "Image directory:"
    )

    print(
        SAVE_DIR
    )

    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()