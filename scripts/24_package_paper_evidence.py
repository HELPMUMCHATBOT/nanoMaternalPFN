#!/usr/bin/env python3
"""Create and checksum an actual evidence bundle; fail if required data absent."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

REQUIRED = (
    "paper_all_methods_summary.csv",
    "paper_completion_full",
    "downstream_v2_full",
    "edge_paper_comparison/profile.json",
    "paper_onnx_memory.json",
)
OPTIONAL = (
    "downstream_full", "archived/downstream_full_20261111.tar.gz",
    "paper_three_expert_full", "adaptive_cpu_profile.json", "paper_verified",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def assemble(root, output):
    root = Path(root)
    output = Path(output)
    for relative in REQUIRED:
        path = root / relative
        if not path.exists() or (path.is_dir() and not any(path.rglob("*"))):
            raise FileNotFoundError(f"Cannot claim bundle completeness: {path}")
    members = {}
    found = []
    for relative in REQUIRED + OPTIONAL:
        path = root / relative
        if not path.exists():
            continue
        found.append(relative)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if file.resolve() == output.resolve() or file.suffix == ".zip":
                continue
            arc = str(file.relative_to(root.parent))
            if arc in members:
                raise RuntimeError("Duplicate archive entry: " + arc)
            members[arc] = file
    if not members:
        raise RuntimeError("No evidence files")
    manifest = {
        "status":"VERIFIED_ARCHIVE_ONLY_NOT_CLINICAL_CONFIRMATION",
        "required_found":list(REQUIRED),
        "optional_found":[p for p in found if p not in REQUIRED],
        "optional_missing":[p for p in OPTIONAL if p not in found],
        "files":[{"path":arc,"sha256":sha(path),"bytes":path.stat().st_size}
                 for arc,path in sorted(members.items())],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for arc,path in sorted(members.items()):
            archive.write(path, arcname=arc)
        archive.writestr("EVIDENCE_SHA256_MANIFEST.json",
                         json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
        stored = json.loads(archive.read("EVIDENCE_SHA256_MANIFEST.json"))
        if len(stored["files"]) != len(members):
            raise RuntimeError("Archive manifest count mismatch")
    return {"bundle":str(output),"sha256":sha(output),
            "files":len(members),"optional_missing":manifest["optional_missing"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/results"))
    p.add_argument("--output",type=Path,
                   default=Path("artifacts/results/paper_review_bundle.zip"))
    args=p.parse_args()
    print(json.dumps(assemble(args.root,args.output),indent=2))


if __name__ == "__main__":
    main()
