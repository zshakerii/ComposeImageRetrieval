import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

JSON_FILE = r"C:\Users\user\Desktop\Python\ImageRetrieval\nlvr\nlvr2\data\dev.json"

SAVE_DIR = r"C:\Users\user\Desktop\Python\ImageRetrieval\nlvr\nlvr2\images\dev"

MAX_WORKERS = 8

TIMEOUT = 20

MAX_RETRIES = 3


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    )
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# LOAD DEV.JSON
# ============================================================

def load_examples():

    examples = []

    with open(
        JSON_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:

                example = json.loads(line)

                examples.append(example)

            except json.JSONDecodeError as ex:

                print(
                    f"[ERROR] Invalid JSON "
                    f"at line {line_number}: {ex}"
                )

    return examples


# ============================================================
# BUILD IMAGE LIST
# ============================================================

def build_image_list(examples):

    images = {}

    for example in examples:

        identifier = example.get("identifier")

        if not identifier:
            continue

        # ----------------------------------------------------
        # Example:
        #
        # dev-149-2-0
        #
        # We need:
        #
        # dev-149-2
        # ----------------------------------------------------

        parts = identifier.split("-")

        if len(parts) < 3:

            print(
                f"[WARNING] Invalid identifier: "
                f"{identifier}"
            )

            continue

        image_id = "-".join(parts[:3])

        # ----------------------------------------------------
        # URLs
        # ----------------------------------------------------

        left_url = example.get("left_url")

        right_url = example.get("right_url")

        # ----------------------------------------------------
        # Image 0
        # ----------------------------------------------------

        if left_url:

            filename = f"{image_id}-img0.png"

            if filename not in images:

                images[filename] = left_url

        # ----------------------------------------------------
        # Image 1
        # ----------------------------------------------------

        if right_url:

            filename = f"{image_id}-img1.png"

            if filename not in images:

                images[filename] = right_url

    return images


# ============================================================
# DOWNLOAD ONE IMAGE
# ============================================================

def download_image(filename, url):

    final_path = os.path.join(
        SAVE_DIR,
        filename
    )

    # ========================================================
    # VERY IMPORTANT:
    # CHECK FILE BEFORE MAKING HTTP REQUEST
    # ========================================================

    if os.path.isfile(final_path):

        return {
            "status": "exists",
            "filename": filename,
            "url": url
        }

    # ========================================================
    # TEMP FILE
    # ========================================================

    temp_path = final_path + ".tmp"

    last_error = None

    # ========================================================
    # RETRY
    # ========================================================

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            # ------------------------------------------------
            # Remove previous temporary file
            # ------------------------------------------------

            if os.path.exists(temp_path):

                try:

                    os.remove(temp_path)

                except Exception:
                    pass

            # ------------------------------------------------
            # HTTP REQUEST
            # ------------------------------------------------

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=TIMEOUT,
                stream=True,
                allow_redirects=True
            )

            # ------------------------------------------------
            # HTTP ERROR
            # ------------------------------------------------

            response.raise_for_status()

            # ------------------------------------------------
            # SAVE IMAGE
            # ------------------------------------------------

            with open(
                temp_path,
                "wb"
            ) as file:

                for chunk in response.iter_content(
                    chunk_size=64 * 1024
                ):

                    if chunk:

                        file.write(chunk)

            # ------------------------------------------------
            # CHECK DOWNLOADED FILE
            # ------------------------------------------------

            if not os.path.exists(temp_path):

                raise Exception(
                    "Downloaded file does not exist"
                )

            if os.path.getsize(temp_path) == 0:

                raise Exception(
                    "Downloaded file is empty"
                )

            # ------------------------------------------------
            # MOVE TEMP FILE TO FINAL FILE
            # ------------------------------------------------

            os.replace(
                temp_path,
                final_path
            )

            return {
                "status": "downloaded",
                "filename": filename,
                "url": url
            }

        except Exception as ex:

            last_error = str(ex)

            # ------------------------------------------------
            # Retry
            # ------------------------------------------------

            if attempt < MAX_RETRIES:

                time.sleep(1)

    # ========================================================
    # DOWNLOAD FAILED
    # ========================================================

    return {
        "status": "failed",
        "filename": filename,
        "url": url,
        "error": last_error
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("NLVR2 DEV IMAGE DOWNLOADER")
    print("=" * 70)

    print()
    print("JSON file:")
    print(JSON_FILE)

    print()
    print("Image directory:")
    print(SAVE_DIR)

    print()
    print("=" * 70)

    # ========================================================
    # CHECK JSON FILE
    # ========================================================

    if not os.path.isfile(JSON_FILE):

        print()
        print(
            "[ERROR] dev.json not found:"
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
        f"Unique images found: {len(images)}"
    )

    # ========================================================
    # CHECK EXISTING FILES
    # ========================================================

    existing_images = []

    images_to_download = {}

    for filename, url in images.items():

        file_path = os.path.join(
            SAVE_DIR,
            filename
        )

        # ----------------------------------------------------
        # File already exists
        # ----------------------------------------------------

        if os.path.isfile(file_path):

            existing_images.append(
                filename
            )

        # ----------------------------------------------------
        # File does not exist
        # ----------------------------------------------------

        else:

            images_to_download[filename] = url

    # ========================================================
    # PRINT STATISTICS
    # ========================================================

    print()
    print("-" * 70)

    print(
        f"Total images     : {len(images)}"
    )

    print(
        f"Already exists   : {len(existing_images)}"
    )

    print(
        f"Need download    : {len(images_to_download)}"
    )

    print("-" * 70)

    # ========================================================
    # NOTHING TO DOWNLOAD
    # ========================================================

    if not images_to_download:

        print()
        print(
            "All images already exist."
        )

        print(
            "No HTTP requests were made."
        )

        print()
        print("=" * 70)

        return

    # ========================================================
    # DOWNLOAD
    # ========================================================

    results = []

    print()
    print(
        "Starting download..."
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {}

        for filename, url in images_to_download.items():

            future = executor.submit(
                download_image,
                filename,
                url
            )

            futures[future] = filename

        # ----------------------------------------------------
        # Progress bar
        # ----------------------------------------------------

        with tqdm(
            total=len(futures),
            desc="Downloading",
            unit="image"
        ) as progress:

            for future in as_completed(futures):

                filename = futures[future]

                try:

                    result = future.result()

                except Exception as ex:

                    result = {
                        "status": "failed",
                        "filename": filename,
                        "error": str(ex)
                    }

                results.append(result)

                progress.update(1)

    # ========================================================
    # STATISTICS
    # ========================================================

    downloaded_count = sum(
        1
        for result in results
        if result["status"] == "downloaded"
    )

    failed_results = [
        result
        for result in results
        if result["status"] == "failed"
    ]

    # ========================================================
    # SAVE FAILED DOWNLOADS
    # ========================================================

    failed_file = os.path.join(
        SAVE_DIR,
        "failed_downloads.txt"
    )

    with open(
        failed_file,
        "w",
        encoding="utf-8"
    ) as file:

        for result in failed_results:

            file.write(
                f"{result.get('filename', '')}\t"
                f"{result.get('url', '')}\t"
                f"{result.get('error', '')}\n"
            )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("DOWNLOAD FINISHED")
    print("=" * 70)

    print()
    print(
        f"Total images     : {len(images)}"
    )

    print(
        f"Already existed  : {len(existing_images)}"
    )

    print(
        f"Downloaded       : {downloaded_count}"
    )

    print(
        f"Failed           : {len(failed_results)}"
    )

    print()
    print(
        "Image directory:"
    )

    print(
        SAVE_DIR
    )

    # ========================================================
    # FAILED FILES
    # ========================================================

    if failed_results:

        print()
        print(
            "Failed downloads:"
        )

        print(
            failed_file
        )

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()