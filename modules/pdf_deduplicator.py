"""
PDF Deduplication Helper
Scans a folder of PDFs, computes SHA256 hashes, and identifies duplicates.

Usage:
    python modules/pdf_deduplicator.py
"""

import hashlib
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def get_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash for a PDF file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicates(pdf_directory: str, cache_path: str = "processed_hashes.json"):
    """
    Compute hashes for all PDFs and find duplicates.

    Args:
        pdf_directory (str): Directory containing PDFs (nested folders supported)
        cache_path (str): Path to store or read hash cache
    """
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.rglob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return

    logger.info(f"📂 Found {len(pdf_files)} PDF files (including nested folders)")

    # Load existing hash cache if available
    cache_file = Path(cache_path)
    if cache_file.exists():
        processed = json.loads(cache_file.read_text())
        logger.info(f"Loaded {len(processed)} cached file hashes.")
    else:
        processed = {}

    duplicates = []
    new_hashes = {}

    for pdf_file in pdf_files:
        file_hash = get_file_hash(pdf_file)
        if file_hash in processed or file_hash in new_hashes:
            logger.info(f"⚠️ Duplicate detected: {pdf_file.name}")
            duplicates.append(str(pdf_file))
        else:
            new_hashes[file_hash] = str(pdf_file)

    # Merge and save updated cache
    processed.update(new_hashes)
    cache_file.write_text(json.dumps(processed, indent=2))
    logger.info(f"✅ Saved updated hash log to {cache_file}")

    logger.info("=" * 60)
    logger.info(f"📊 Deduplication Summary")
    logger.info("=" * 60)
    logger.info(f"Total PDFs scanned: {len(pdf_files)}")
    logger.info(f"Unique files: {len(processed)}")
    logger.info(f"Duplicates found: {len(duplicates)}")
    logger.info("=" * 60)

    if duplicates:
        dup_log = pdf_dir / "duplicate_list.txt"
        dup_log.write_text("\n".join(duplicates))
        logger.info(f"📝 Duplicate list saved to {dup_log}")


if __name__ == "__main__":
    # 🔧 Adjust this to match your structure
    base_dir = Path("/Users/yashasvireddymusku/Desktop/backups/rag-dev_backup_20251028_133704")
    pdf_dir = base_dir / "data" / "raw_pdfs"
    cache_file = base_dir / "data" / "processed" / "processed_hashes.json"

    find_duplicates(str(pdf_dir), str(cache_file))
