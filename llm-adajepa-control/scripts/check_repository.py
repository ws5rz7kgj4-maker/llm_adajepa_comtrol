"""Check local documentation links, Python syntax and obvious credential patterns."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", "runs"}


def main():
    files = [p for p in ROOT.rglob("*") if p.is_file() and not (set(p.relative_to(ROOT).parts) & IGNORED)]
    issues = []
    python_count = 0
    link_count = 0
    patterns = [
        r"hf_[A-Za-z0-9]{24,}",
        r"gh[pousr]_[A-Za-z0-9]{25,}",
        r"sk-[A-Za-z0-9]{30,}",
        r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----",
    ]
    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path.suffix == ".py":
            ast.parse(content, filename=str(path.relative_to(ROOT)))
            python_count += 1
        if path.suffix == ".md":
            for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)\)", content):
                if re.match(r"(https?://|mailto:|#)", target):
                    continue
                local = target.split("#", 1)[0]
                if local:
                    link_count += 1
                    if not (path.parent / local).exists():
                        issues.append(f"Missing link in {path.relative_to(ROOT)}: {target}")
        if any(re.search(pattern, content) for pattern in patterns):
            issues.append(f"Credential-like text: {path.relative_to(ROOT)}")
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"PASS: {python_count} Python files parsed; {link_count} local links resolve.")
    print("No matching credential patterns found. External links and credential absence are not exhaustively verified.")


if __name__ == "__main__":
    main()
