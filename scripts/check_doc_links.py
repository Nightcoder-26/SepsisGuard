# -*- coding: utf-8 -*-
"""
Documentation Link Integrity Verification Script (Phase 12)
Parses Markdown files across the repository and verifies internal relative file link existence.
"""

import os
import sys
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DOC_FILES = [
    "README.md",
    "ARCHITECTURE.md",
    "DEPLOYMENT.md",
    "model/MODEL_CARD.md",
    "data/DATASET_CARD.md",
    "validation/EXTERNAL_DATASET_CARD.md",
    "validation/EXTERNAL_VALIDATION_DATA_CARD.md",
    "validation/GENERALIZATION_REPORT.md",
    "validation/subgroup_analysis.md"
]

LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

def check_doc_links():
    print("=" * 70)
    print("SEPSISGUARD PHASE 12 - DOCUMENTATION LINK INTEGRITY CHECK")
    print("=" * 70)

    broken_links = 0
    total_links = 0

    for doc_rel in DOC_FILES:
        doc_path = os.path.join(PROJECT_ROOT, doc_rel)
        if not os.path.exists(doc_path):
            print(f"[SKIP] {doc_rel} does not exist.")
            continue

        doc_dir = os.path.dirname(doc_path)
        content = open(doc_path, "r", encoding="utf-8", errors="ignore").read()
        links = LINK_PATTERN.findall(content)

        for label, url in links:
            # Ignore HTTP(S) external links or anchor-only links
            if url.startswith("http://") or url.startswith("https://") or url.startswith("#"):
                continue

            total_links += 1
            # Clean file:// URI prefix if present
            clean_url = url.replace("file:///", "").replace("file://", "")
            # Remove line number fragments e.g. #L10
            clean_url = clean_url.split("#")[0]

            if os.path.isabs(clean_url):
                target_path = clean_url
            else:
                target_path = os.path.normpath(os.path.join(doc_dir, clean_url))

            if not os.path.exists(target_path):
                print(f"[BROKEN LINK] In {doc_rel}: '{label}' -> '{url}' (Target not found: {target_path})")
                broken_links += 1

    print("-" * 70)
    print(f"[*] Verified {total_links} internal documentation links across project docs.")

    if broken_links > 0:
        print(f"[FAIL] Found {broken_links} broken documentation links!")
        return False

    print("[OK] ALL DOCUMENTATION LINKS VERIFIED INTAC T & VALID.")
    return True

if __name__ == '__main__':
    success = check_doc_links()
    sys.exit(0 if success else 1)
