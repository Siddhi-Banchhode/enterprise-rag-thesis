# scripts/download_datasets.py
import os, sys, json, subprocess
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import config
from datasets import load_dataset
import pandas as pd

DATASETS_DIR = Path(config.DATASETS_DIR)
CUAD_DIR  = DATASETS_DIR / "cuad"
SQUAD_DIR = DATASETS_DIR / "squad2"
PDFQA_DIR = DATASETS_DIR / "pdfqa"
for d in [CUAD_DIR, SQUAD_DIR, PDFQA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def download_pdfqa():
    target = PDFQA_DIR / "pdfQA-Benchmark"
    if target.exists():
        print(f"✅ pdfQA-Benchmark already exists")
        return str(target)

    print("Installing git-lfs...")
    os.system("brew install git-lfs 2>/dev/null; git lfs install")

    print("Cloning pdfQA-Benchmark (includes PDFs, may take a few minutes)...")
    cmd = f"git clone https://huggingface.co/datasets/pdfqa/pdfQA-Benchmark {target}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("Git clone failed, trying huggingface_hub instead...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="pdfqa/pdfQA-Benchmark", repo_type="dataset",
                          local_dir=str(target), ignore_patterns=["*.git*"])

    print(f"✅ pdfQA-Benchmark ready at: {target}")
    return str(target)


def download_cuad(max_samples=200):
    """
    Downloads CUAD-QA — the SQuAD-formatted version of CUAD with
    proper question/context/answers fields (not the raw text-only version).
    """
    save_path = CUAD_DIR / "cuad_eval.json"
    if save_path.exists():
        with open(save_path) as f: return json.load(f)

    print("Downloading CUAD (cuad-qa, SQuAD-formatted)...")
    dataset = load_dataset(
        "theatticusproject/cuad-qa",
        split="train",
        trust_remote_code=True
    )

    qa_pairs = []
    for item in dataset:
        answers = item.get("answers", {}).get("text", [])
        if not answers:
            continue

        qa_pairs.append({
            "question": item.get("question", "").strip(),
            "answer":   answers[0].strip(),
            "context":  item.get("context", "")[:600],
            "title":    item.get("title", "unknown"),
            "dataset":  "CUAD"
        })

        if len(qa_pairs) >= max_samples:
            break

    with open(save_path, "w") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

    print(f"✅ CUAD saved: {len(qa_pairs)} pairs")
    return qa_pairs


def download_squad2(max_each=50):
    save_path = SQUAD_DIR / "squad2_eval.json"
    if save_path.exists():
        with open(save_path) as f: return json.load(f)

    print("Downloading SQuAD 2.0...")
    dataset = load_dataset("rajpurkar/squad_v2", split="validation")

    answerable, unanswerable = [], []
    for item in dataset:
        ans = item.get("answers", {}).get("text", [])
        entry = {
            "question": item.get("question", "").strip(),
            "context": item.get("context", "")[:600],
            "dataset": "SQuAD2"
        }
        if ans and len(answerable) < max_each:
            entry["answer"] = ans[0].strip()
            entry["is_answerable"] = True
            answerable.append(entry)
        elif not ans and len(unanswerable) < max_each:
            entry["answer"] = None
            entry["is_answerable"] = False
            unanswerable.append(entry)
        if len(answerable) >= max_each and len(unanswerable) >= max_each:
            break

    all_pairs = answerable + unanswerable
    with open(save_path, "w") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    print(f"✅ SQuAD 2.0 saved: {len(answerable)} answerable + "
          f"{len(unanswerable)} unanswerable")
    return all_pairs


def copy_pdfs_to_main_folder(pdfqa_root, max_pdfs=15, max_per_domain=3):
    """
    Copies a small, representative sample of PDFs into data/pdfs/
    for the working RAG corpus. Does NOT copy all 4,622 files (25GB) —
    only a manageable sample for indexing and demonstration.
    """
    import shutil

    pdf_dir = Path(config.PDF_DIR)
    source = Path(pdfqa_root) / "real-pdfQA" / "01.2_Input_Files_PDF"

    if not source.exists():
        print(f"⚠ No PDFs found at {source}")
        print(f"  Add PDFs manually to data/pdfs/")
        return

    count = 0
    domain_counts = {}

    domains = [d for d in source.iterdir() if d.is_dir()]
    print(f"Found {len(domains)} domains: {[d.name for d in domains]}")

    for domain_dir in domains:
        if count >= max_pdfs:
            break

        domain_counts[domain_dir.name] = 0
        pdfs_in_domain = list(domain_dir.glob("*.pdf"))

        for pdf in pdfs_in_domain:
            if count >= max_pdfs or domain_counts[domain_dir.name] >= max_per_domain:
                break

            dest = pdf_dir / f"{domain_dir.name}_{pdf.name}"
            if not dest.exists():
                shutil.copy2(pdf, dest)
                count += 1
                domain_counts[domain_dir.name] += 1

    print(f"✅ Copied {count} PDFs to {pdf_dir} (sampled across "
          f"{len([d for d in domain_counts.values() if d > 0])} domains)")
    print(f"   Breakdown: {domain_counts}")
    print(f"   Note: Full dataset has 4,622 PDFs (25GB) at {source}")
    print(f"   Only a sample was copied to keep the working corpus manageable.")


if __name__ == "__main__":
    print("Starting dataset downloads...\n")
    pdfqa_root = download_pdfqa()
    cuad  = download_cuad()
    squad = download_squad2()
    copy_pdfs_to_main_folder(pdfqa_root, max_pdfs=15, max_per_domain=3)   

    print("\n" + "="*50)
    print(f"CUAD: {len(cuad)} pairs | SQuAD2: {len(squad)} pairs")
    print(f"PDFs in data/pdfs: "
          f"{len(list(Path(config.PDF_DIR).glob('*.pdf')))} files")
    print("✅ All datasets ready")