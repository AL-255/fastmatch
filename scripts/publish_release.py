"""Publish all validated native artifacts from a completed tagged workflow.

Download/upload one artifact at a time so GPU assets do not fill the publisher
runner's disk. New releases stay draft until every platform's assets are uploaded.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def publish(repo, tag, run_id):
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", tag):
        raise ValueError("Not a release version tag")
    run = json.loads(gh("api", f"repos/{repo}/actions/runs/{run_id}"))
    artifacts = []
    page = 1
    while True:
        data = json.loads(gh("api", f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"))
        artifacts.extend(a for a in data["artifacts"] if a["name"].startswith("FastMatch-") and not a["expired"])
        if len(data["artifacts"]) < 100:
            break
        page += 1
    expected = {"FastMatch-windows-x64-cpu", "FastMatch-windows-x64-cuda-cu128",
                "FastMatch-linux-x64-cpu", "FastMatch-linux-x64-cuda-cu128",
                "FastMatch-linux-x64-rocm7.2", "FastMatch-macos-arm64-mps"}
    if {a["name"] for a in artifacts} != expected:
        raise RuntimeError("Missing validated build artifacts; refusing to publish a partial release")
    exists = subprocess.run(["gh", "release", "view", tag, "--repo", repo], capture_output=True).returncode == 0
    if not exists:
        gh("release", "create", tag, "--repo", repo, "--verify-tag", "--draft", "--generate-notes",
           "--title", f"FastMatch {tag}", "--target", run["head_sha"])
    for artifact in sorted(artifacts, key=lambda a: a["name"]):
        with tempfile.TemporaryDirectory(prefix="fastmatch-assets-") as temporary:
            gh("run", "download", str(run_id), "--repo", repo, "--name", artifact["name"], "--dir", temporary)
            files = sorted(str(p) for p in Path(temporary).iterdir() if p.is_file())
            if not files:
                raise RuntimeError(f"Empty artifact: {artifact['name']}")
            gh("release", "upload", tag, "--repo", repo, "--clobber", *files)
    gh("release", "upload", tag, "--repo", repo, "--clobber",
       "scripts/assemble_release.py", "scripts/assemble_release.ps1")
    gh("release", "edit", tag, "--repo", repo, "--draft=false",
       "--prerelease=true" if "-" in tag else "--prerelease=false")
    print(gh("release", "view", tag, "--repo", repo, "--json", "url", "--jq", ".url"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    publish(args.repo, args.tag, args.run_id)
