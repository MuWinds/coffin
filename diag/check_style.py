"""Comment-convention gate for this repo: the executable form of ``AGENTS.md``.

Module: ``coffin/diag``. Read-only tool run by hand and by the refactor loop.
It answers two questions that eyeballing 20k lines cannot:

* **Did the edit break code?** Compare against ``git HEAD`` with docstrings
  stripped (Python AST) or comments stripped (CUDA/C/C++ token stream).
  A comment refactor is allowed to move exactly those two things.
* **Does a comment meet the standard?** Non-English prose, missing space after
  the marker, ASCII dividers, Sphinx field lists, missing file-header sections,
  missing ``Args`` / ``Returns`` / ``Raises``, and the stock phrases that mark
  generated filler.

Constraints:
    * Read-only: never writes, never fixes. Verdict goes to stdout and the exit
      code (0 = pass), so it can gate a batch of edits.
    * The baseline is ``git HEAD``, so start a refactor from a clean worktree;
      otherwise the baseline already contains the edits you are checking.
    * Comparison always happens after comments are stripped. A changed code
      token is an error, never something to wave through by hand.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

#: Prose must be English. A CJK ideograph outside a quoted sample means the
#: comment was not rewritten to the standard.
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

#: A CJK run inside backticks or quotes is sample data (a prompt fragment, a
#: protocol example) and is allowed to stay non-ASCII.
CJK_IN_QUOTES_RE = re.compile(r"(`[^`]*[\u4e00-\u9fff][^`]*`|\"[^\"]*[\u4e00-\u9fff][^\"]*\")")

#: Marker must be followed by exactly one space (or nothing, or a `!`/`:` form).
MARKER_SPACE_RE = re.compile(r"^\s*(?:#|//)(?![ \t])(?![!:\-=*#/])")

#: ASCII rules: three or more of the same symbol across a whole line.
DECORATION_RE = re.compile(r"^\s*(?:(?:#|//|/\*|\*)\s*)?([-=*_~#+.])\1{2,}\s*\*?/?\s*$")

#: Sphinx field lists are not the house style.
SPHINX_FIELD_RE = re.compile(r":(param|parameter|returns?|raises?|yields?|type)\b")

#: Google Style sections. Anything else (NumPy `Parameters`, `----------`) is a miss.
GOOGLE_SECTION_RE = re.compile(
    r"^\s*(Args|Arguments|Returns|Raises|Yields|Example|Examples|Note|Attributes)\s*:",
    re.M,
)

#: File-header keywords required by AGENTS.md section 4.
HEADER_SECTION_KEYS = ("Module:", "Constraints:")

#: Filler phrases AGENTS.md section 7 bans. Each one is either redundant or
#: padded, and the sentence reads better without it.
FILLER_PHRASES = (
    "note that",
    "it is important to note",
    "it should be noted",
    "this function is responsible for",
    "this method is responsible for",
    "here we",
    "in order to",
    "leverage",
    "utilize",
    "utilise",
    "seamlessly",
    "comprehensive",
    "robust",
    "simply",
    "basically",
    "as you can see",
    "please note",
    "make sure to",
)

#: Tool directives are matched by literal string, so they keep their English.
TOOL_DIRECTIVE_RE = re.compile(
    r"^\s*(#|//)\s*(noqa|type:|pragma:|pylint:|isort:|fmt:|nosec|mypy:|!|-\*-|coding[:=])"
)

PY_SUFFIXES = (".py",)
C_SUFFIXES = (".cu", ".cuh", ".c", ".h", ".hpp", ".cc", ".cpp")

#: Tests and probes are not library API, so AGENTS.md section 5 does not demand
#: the four sections there. Language and formatting still apply.
RELAXED_DIRS = ("tests", "diag")


class Issue:
    """One finding. ``severity == "error"`` fails the run; ``"warn"`` does not."""

    __slots__ = ("path", "line", "severity", "code", "message")

    def __init__(self, path: str, line: int, severity: str, code: str, message: str) -> None:
        self.path = path
        self.line = line
        self.severity = severity
        self.code = code
        self.message = message

    def render(self) -> str:
        """Format as one line, ``ERROR path:line [code] message``, for grep and IDE jump."""
        return f"{self.severity.upper():5s} {self.path}:{self.line} [{self.code}] {self.message}"


def _git(repo: Path, *args: str) -> str | None:
    """Run one read-only git command.

    Args:
        repo: Repository root (``coffin/`` is its own repo).
        *args: Arguments passed to ``git``.

    Returns:
        Standard output, or ``None`` when the command fails — which is normal
        for a file that does not exist in ``HEAD`` yet (a new file has no
        baseline, so there is nothing to compare and nothing to report).
    """
    try:
        done = subprocess.run(
            ["git", *args], cwd=str(repo), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return done.stdout


def _repo_root(start: Path) -> Path | None:
    """Walk up to the nearest ``.git`` so the diff baseline is the right repo."""
    for cand in [start, *start.parents]:
        if (cand / ".git").exists():
            return cand
    return None


def _head_text(repo: Path, path: Path) -> str | None:
    """Return the ``git HEAD`` version of a file, or ``None`` for a new file."""
    try:
        rel = path.resolve().relative_to(repo.resolve())
    except ValueError:
        return None
    return _git(repo, "show", f"HEAD:{rel.as_posix()}")


class _DocstringStripper(ast.NodeTransformer):
    """Drop the leading string statement of modules, classes and functions.

    Rewriting a docstring is the whole point of this refactor, so AST equality
    has to mean "equal once docstrings are gone". An empty body is refilled
    with ``Pass`` so both sides keep the same shape.
    """

    def _strip(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:] or [ast.Pass()]
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        return self._strip(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._strip(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._strip(node)


def _dump(src: str) -> str | None:
    """Normalise source to position-free AST text, or ``None`` on a syntax error.

    ``feature_version`` defaults to the running interpreter; ``include_attributes``
    is off so moving code within a file does not register as a change.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def _strip_c_comments(src: str) -> str:
    """Remove ``//`` and ``/* */`` comments while preserving literals.

    A hand-written scanner rather than a regex: ``"http://..."`` inside a string
    would defeat a naive one, and URLs appear in this repo's protocol code.

    Args:
        src: Original C-family source.

    Returns:
        Equivalent source with comments blanked out.
    """
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                if src[i] == "\\" and i + 1 < n:
                    out.append(src[i : i + 2])
                    i += 2
                    continue
                out.append(src[i])
                if src[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            end = src.find("\n", i)
            i = n if end < 0 else end
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            end = src.find("*/", i + 2)
            i = n if end < 0 else end + 2
            out.append(" ")
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_ws(text: str) -> str:
    """Collapse whitespace so comment removal does not itself look like a change."""
    return re.sub(r"\s+", " ", text).strip()


def _iter_py_comments(src: str) -> list[tuple[int, str]]:
    """List ``#`` comments as ``(line, raw_text)``."""
    found: list[tuple[int, str]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                found.append((tok.start[0], tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return found
    return found


def _iter_c_comments(src: str) -> list[tuple[int, str]]:
    """List C-family comment blocks as ``(start_line, raw_text)``.

    A block comment stays one unit: a Doxygen ``@brief`` / ``@param`` group has
    to be judged as a whole.
    """
    found: list[tuple[int, str]] = []
    i = 0
    n = len(src)
    line = 1
    while i < n:
        ch = src[i]
        if ch in ("'", '"'):
            quote = ch
            i += 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == quote:
                    i += 1
                    break
                if src[i] == "\n":
                    line += 1
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            end = src.find("\n", i)
            end = n if end < 0 else end
            found.append((line, src[i:end]))
            i = end
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            end = src.find("*/", i + 2)
            end = n if end < 0 else end + 2
            found.append((line, src[i:end]))
            line += src.count("\n", i, end)
            i = end
            continue
        if ch == "\n":
            line += 1
        i += 1
    return found


def _comment_body(raw: str) -> str:
    """Strip the comment markers, leaving the prose to be judged."""
    text = re.sub(r"^\s*(#|//)", "", raw.strip())
    text = re.sub(r"^\s*(/\*+|\*+/?)\s?", "", text)
    text = text.replace("*/", "")
    text = re.sub(r"^\*+\s?", "", text, flags=re.M)
    return text.strip()


def _prose_without_samples(body: str) -> str:
    """Drop quoted/backticked spans so sample data is not judged as prose."""
    return CJK_IN_QUOTES_RE.sub(" ", body)


def _check_comment_text(
    path_str: str, line: int, raw: str, issues: list[Issue], allow_c_marker: bool
) -> None:
    """Check one comment: language, marker spacing, dividers, filler phrases."""
    body = _comment_body(raw)
    if not body:
        return
    if TOOL_DIRECTIVE_RE.match(raw):
        return

    if DECORATION_RE.match(raw.strip()) or DECORATION_RE.match(body):
        issues.append(
            Issue(path_str, line, "error", "STYLE001",
                  "ASCII divider; use a plain `# label` section header instead")
        )

    stripped = raw.lstrip()
    marker_ok = stripped.startswith("#") or (allow_c_marker and stripped.startswith("//"))
    if marker_ok and MARKER_SPACE_RE.match(stripped):
        issues.append(
            Issue(path_str, line, "error", "STYLE002",
                  "missing space between the marker and the text: `# text`")
        )

    prose = _prose_without_samples(body)
    if CJK_RE.search(prose):
        issues.append(
            Issue(path_str, line, "error", "STYLE003",
                  f"non-English prose; rewrite in English: {body[:60]}")
        )

    lowered = body.lower()
    for phrase in FILLER_PHRASES:
        if phrase in lowered:
            issues.append(
                Issue(path_str, line, "warn", "STYLE005",
                      f"filler phrase “{phrase}”; state the fact instead")
            )
            break


def _own_nodes(func: ast.AST):
    """Walk a function's own statements without entering nested definitions.

    A nested function carries its own contract, so the outer one should not be
    blamed for its ``raise`` or its ``return``.
    """
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _param_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """List parameter names, minus ``self`` / ``cls`` and the bare ``*`` marker."""
    args = node.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    names = [n for n in names if n not in ("self", "cls")]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _returns_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the function returns something other than a bare ``None``."""
    for inner in _own_nodes(node):
        if isinstance(inner, ast.Return) and inner.value is not None:
            if isinstance(inner.value, ast.Constant) and inner.value.value is None:
                continue
            return True
    return False


def _raises(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the function's own body raises."""
    return any(isinstance(inner, ast.Raise) for inner in _own_nodes(node))


def _is_trivial(node: ast.AST) -> bool:
    """Whether a function body is a one-expression wrapper that explains itself.

    A property that returns ``len(self.token_ids)`` or a delegating
    ``__getitem__`` needs no docstring: the signature already says everything,
    and demanding one only produces the restating-the-code comments that
    AGENTS.md section 2 deletes. Anything with a branch, a raise, a loop or more
    than two statements does need one.
    """
    body = getattr(node, "body", None)
    if not body or len(body) > 2:
        return False
    for stmt in body:
        if isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Raise)):
            return False
    return True


def _check_docstring(
    path_str: str, node: ast.AST, doc: str | None, relaxed: bool, issues: list[Issue]
) -> None:
    """Check one class/function docstring against the Google Style sections."""
    name = getattr(node, "name", "<module>")
    line = getattr(node, "lineno", 1)
    is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    if doc is None:
        if _is_trivial(node):
            return
        severity = "warn" if relaxed or name.startswith("_") else "error"
        issues.append(
            Issue(path_str, line, severity, "STYLE010",
                  f"`{name}` has no docstring; say what it is for and what constrains it")
        )
        return

    if SPHINX_FIELD_RE.search(doc):
        issues.append(
            Issue(path_str, line, "error", "STYLE011",
                  f"`{name}` uses Sphinx field lists; this repo is Google Style")
        )

    if CJK_RE.search(_prose_without_samples(doc)):
        issues.append(
            Issue(path_str, line, "error", "STYLE003", f"`{name}` docstring is not English")
        )

    lowered = doc.lower()
    for phrase in FILLER_PHRASES:
        if phrase in lowered:
            issues.append(
                Issue(path_str, line, "warn", "STYLE005",
                      f"`{name}` docstring contains filler “{phrase}”")
            )
            break

    if relaxed:
        return

    sections = {m.group(1) for m in GOOGLE_SECTION_RE.finditer(doc)}
    if is_func:
        params = _param_names(node)
        if params and not sections & {"Args", "Arguments"}:
            issues.append(
                Issue(path_str, line, "warn", "STYLE012",
                      f"`{name}` takes {params[:4]} but has no `Args:` section")
            )
        if _returns_value(node) and "Returns" not in sections:
            issues.append(
                Issue(path_str, line, "warn", "STYLE013",
                      f"`{name}` returns a value but has no `Returns:` section")
            )
        if _raises(node) and "Raises" not in sections:
            issues.append(
                Issue(path_str, line, "warn", "STYLE014",
                      f"`{name}` raises but has no `Raises:` section")
            )


def check_ast(repo: Path, path: Path) -> list[Issue]:
    """Verify a Python file changed only docstrings and comments versus ``HEAD``.

    Args:
        repo: Repository root.
        path: The ``.py`` file to check.

    Returns:
        Findings; empty means the code structure matches the baseline exactly.
    """
    path_str = str(path)
    head = _head_text(repo, path)
    if head is None:
        return []
    try:
        current = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [Issue(path_str, 0, "error", "AST900", f"unreadable: {exc}")]
    before = _dump(head)
    after = _dump(current)
    if before is None or after is None:
        return [Issue(path_str, 0, "error", "AST901", "parse failed, cannot compare")]
    if before != after:
        return [
            Issue(path_str, 0, "error", "AST001",
                  "code changed: a comment refactor may only touch docstrings and `#` comments")
        ]
    return []


def check_c_comments(repo: Path, path: Path) -> list[Issue]:
    """Verify a C-family file changed only comments versus ``HEAD``.

    Args:
        repo: Repository root.
        path: The ``.cu`` / ``.c`` / ``.h`` / ``.cpp`` file to check.

    Returns:
        Findings; empty means every code token is untouched.
    """
    path_str = str(path)
    head = _head_text(repo, path)
    if head is None:
        return []
    try:
        current = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [Issue(path_str, 0, "error", "AST900", f"unreadable: {exc}")]
    before = _normalize_ws(_strip_c_comments(head))
    after = _normalize_ws(_strip_c_comments(current))
    if before != after:
        return [
            Issue(path_str, 0, "error", "AST002",
                  "code changed after stripping comments: only comment text may move")
        ]
    return []


def check_py_style(path: Path, relaxed: bool) -> list[Issue]:
    """Check one Python file's comments and docstrings.

    Args:
        path: The ``.py`` file to check.
        relaxed: ``True`` under ``tests/`` / ``diag/``, where the four sections are
            not required but language and formatting still are.

    Returns:
        Findings for the file.
    """
    path_str = str(path)
    issues: list[Issue] = []
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [Issue(path_str, 0, "error", "AST900", f"unreadable: {exc}")]

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [Issue(path_str, exc.lineno or 0, "error", "AST902", f"syntax error: {exc.msg}")]

    module_doc = ast.get_docstring(tree, clean=False)
    if module_doc is None:
        issues.append(
            Issue(path_str, 1, "error", "STYLE020",
                  "no file header docstring: responsibility / module / constraints")
        )
    else:
        if CJK_RE.search(_prose_without_samples(module_doc)):
            issues.append(
                Issue(path_str, 1, "error", "STYLE003", "file header is not English")
            )
        for key in HEADER_SECTION_KEYS:
            if key not in module_doc:
                issues.append(
                    Issue(path_str, 1, "warn", "STYLE021",
                          f"file header has no “{key}” section")
                )

    for line, raw in _iter_py_comments(src):
        _check_comment_text(path_str, line, raw, issues, allow_c_marker=False)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _check_docstring(path_str, node, ast.get_docstring(node, clean=False), relaxed, issues)
    return issues


def check_c_style(path: Path) -> list[Issue]:
    """Check one C-family file's comments.

    Args:
        path: The file to check.

    Returns:
        Findings. A missing ``@file`` / ``@brief`` on the first block is only a
        warning: a short header with no API may not need the full banner.
    """
    path_str = str(path)
    issues: list[Issue] = []
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [Issue(path_str, 0, "error", "AST900", f"unreadable: {exc}")]

    blocks = _iter_c_comments(src)
    for line, raw in blocks:
        _check_comment_text(path_str, line, raw, issues, allow_c_marker=True)

    if blocks:
        first = blocks[0][1]
        if "@file" not in first and "@brief" not in first:
            issues.append(
                Issue(path_str, blocks[0][0], "warn", "STYLE022",
                      "top comment has no Doxygen `@file` / `@brief`")
            )
    return issues


def _iter_targets(root: Path, suffixes: tuple[str, ...]):
    """Yield matching files under ``root``, skipping caches and VCS metadata."""
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in ("__pycache__", ".git", ".vs") for part in path.parts):
            continue
        yield path


def _print_summary(issues: list[Issue], files: int) -> None:
    """Print the tail summary: totals plus the files with the most work left."""
    errors = sum(1 for i in issues if i.severity == "error")
    warns = len(issues) - errors
    print(f"\nscanned {files} files: {errors} error(s), {warns} warning(s)")
    if not issues:
        return
    per_file: dict[str, list[Issue]] = {}
    for issue in issues:
        per_file.setdefault(issue.path, []).append(issue)
    ranked = sorted(per_file.items(), key=lambda kv: -len(kv[1]))
    print("\ntop 15 files by remaining issues:")
    for name, items in ranked[:15]:
        print(f"  {len(items):4d}  {name}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments to parse; ``None`` reads ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 when there is no error, 1 when there is.
    """
    parser = argparse.ArgumentParser(description="coffin comment-convention gate")
    parser.add_argument("paths", nargs="*", help="directories or files, default cwd")
    parser.add_argument("--ast", action="store_true", help="only check Python AST equality")
    parser.add_argument("--style", action="store_true", help="only check comment conventions")
    parser.add_argument("--c-comments", action="store_true", help="only check C-family equality")
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--verbose", action="store_true", help="print every finding")
    args = parser.parse_args(argv)

    targets = [Path(p) for p in args.paths] or [Path(".")]
    repo = _repo_root(targets[0].resolve())
    if repo is None:
        print("no .git found, cannot establish a diff baseline", file=sys.stderr)
        return 2

    do_all = not (args.ast or args.style or args.c_comments)
    do_ast = do_all or args.ast
    do_style = do_all or args.style
    do_c = do_all or args.c_comments

    issues: list[Issue] = []
    checked = 0
    for target in targets:
        if target.is_file():
            # A file argument scans exactly that file. Falling back to its
            # parent directory would silently pull in every sibling, which is
            # the opposite of what "check this file I just edited" means.
            root = target.parent
            files = [target] if target.suffix in PY_SUFFIXES + C_SUFFIXES else []
        else:
            root = target
            files = list(_iter_targets(root, PY_SUFFIXES + C_SUFFIXES))
        for path in files:
            checked += 1
            # Judge relaxation from the path relative to the repo, not to the
            # scan root: `--style diag/probe_x.py` must reach the same verdict as
            # `--style .`, or a per-file gate would be stricter than the
            # whole-tree gate and the two would disagree.
            try:
                rel_parts = path.resolve().relative_to(repo.resolve()).parts
            except ValueError:
                rel_parts = path.parts
            relaxed = any(part in RELAXED_DIRS for part in rel_parts)
            if do_ast and path.suffix in PY_SUFFIXES:
                issues.extend(check_ast(repo, path))
            if do_c and path.suffix in C_SUFFIXES:
                issues.extend(check_c_comments(repo, path))
            if do_style:
                if path.suffix in PY_SUFFIXES:
                    issues.extend(check_py_style(path, relaxed))
                else:
                    issues.extend(check_c_style(path))

    if args.verbose:
        for issue in issues:
            print(issue.render())
    _print_summary(issues, checked)

    errors = sum(1 for i in issues if i.severity == "error")
    warns = len(issues) - errors
    if errors or (args.strict and warns):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
