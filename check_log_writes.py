"""
check_log_writes.py — строгий AST-чекер стратегии логирования для
pretrained_stat.ipynb.

Проверяет, что чтение/запись логов эксперимента идёт ТОЛЬКО через методы
декоратора `experiment_log`:

  * запись — через аргумент `log_path`, который подставляет декоратор;
  * чтение — через `<decorated_func>.read_logs(...)` /
    `<decorated_func>.get_log_path(...)`.

Запрещает: чтение через glob; ручные пути к логам (pretrained/,
pretrained_stat/...); свободные вызовы read_logs()/get_log_path();
файловый I/O по литералу внутри декорированных функций; лишние
@experiment_log; inline-fallback определения experiment_log.

Выход: 0 — нарушений нет; 1 — есть нарушения или синтаксические ошибки.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "pretrained_stat.ipynb"

DECORATED_FUNCTIONS = {
    "run_stage1_pretraining",
    "run_stage2_finetuning",
    "run_stage2_for_analysis_epochs",
    "save_screen_output",
}

READERS = {
    "run_stage2_finetuning",
    "run_stage2_for_analysis_epochs",
}

LEGACY_PATH_RE = re.compile(r"pretrained[/\\]")
STAT_PATH_RE = re.compile(r"pretrained_stat[/\\]")


def _is_string_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _io_path_arg_index(node: ast.Call):
    f = node.func

    if isinstance(f, ast.Name):
        if f.id in ("open", "savefig"):
            return 0
        return None

    if isinstance(f, ast.Attribute):
        if isinstance(f.value, ast.Name):
            ns = f.value.id
            if ns == "json":
                if f.attr == "dump":
                    return 1
                if f.attr == "load":
                    return 0
                return None
            if ns == "torch":
                if f.attr == "save":
                    return 1
                if f.attr == "load":
                    return 0
                return None
            if ns == "np":
                if f.attr in ("save", "savez", "load", "loadtxt", "savetxt"):
                    return 0
                return None
        if f.attr in (
            "to_csv", "to_json", "to_pickle",
            "read_csv", "read_json", "read_pickle", "read_text",
            "savefig",
        ):
            return 0
        return None

    return None


class CellChecker(ast.NodeVisitor):
    def __init__(self, cell_index: int):
        self.cell_index = cell_index
        self.issues: list[str] = []
        self.decorated: list[tuple[str, bool, bool]] = []
        self.import_experiment_log = False
        self.inline_fallback: list[str] = []
        self.read_log_owners: set[str] = set()
        self.get_log_owners: set[str] = set()

        self._inside_decorated = False
        self._current_function: str | None = None
        self._log_path_used: set[str] = set()

    def _issue(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.issues.append(f"cell {self.cell_index} (line {line}): {message}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        decorated = False
        root_ok = False
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "experiment_log"
            ):
                decorated = True
                for kw in dec.keywords:
                    if (
                        kw.arg == "root"
                        and isinstance(kw.value, ast.Name)
                        and kw.value.id == "RESULTS_DIR"
                    ):
                        root_ok = True

        has_log_path = any(
            a.arg == "log_path" for a in node.args.args + node.args.kwonlyargs
        )

        if decorated:
            self.decorated.append((node.name, root_ok, has_log_path))

        prev_inside = self._inside_decorated
        prev_func = self._current_function
        self._inside_decorated = decorated
        self._current_function = node.name
        self.generic_visit(node)
        self._inside_decorated = prev_inside
        self._current_function = prev_func

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "experiment_log":
                self._issue(node, "inline fallback `experiment_log = ...` forbidden")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "experiment_log" and any(
            a.name == "experiment_log" for a in node.names
        ):
            self.import_experiment_log = True
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            if LEGACY_PATH_RE.search(node.value):
                self._issue(node, f"manual legacy log path literal: {node.value!r}")
            if STAT_PATH_RE.search(node.value):
                self._issue(node, f"manual pretrained_stat log path literal: {node.value!r}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            isinstance(node.ctx, ast.Load)
            and node.id == "log_path"
            and self._inside_decorated
            and self._current_function is not None
        ):
            self._log_path_used.add(self._current_function)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func

        if (
            isinstance(f, ast.Attribute)
            and f.attr in ("glob", "iglob")
            and isinstance(f.value, ast.Name)
            and f.value.id == "glob"
        ):
            self._issue(node, "glob-based log read forbidden")
        elif isinstance(f, ast.Name) and f.id == "glob":
            self._issue(node, "glob() call forbidden")

        if isinstance(f, ast.Name) and f.id in ("read_logs", "get_log_path"):
            self._issue(
                node,
                f"free-standing {f.id}(...) call (must be on a decorated function)",
            )
        elif isinstance(f, ast.Attribute) and f.attr in ("read_logs", "get_log_path"):
            owner_ok = isinstance(f.value, ast.Name) and f.value.id in DECORATED_FUNCTIONS
            if owner_ok:
                if f.attr == "read_logs":
                    self.read_log_owners.add(f.value.id)
                else:
                    self.get_log_owners.add(f.value.id)
            else:
                owner = getattr(f.value, "id", type(f.value).__name__)
                self._issue(
                    node,
                    f"{f.attr}(...) called on a non-decorated owner {owner!r}",
                )

        if self._inside_decorated:
            idx = _io_path_arg_index(node)
            if idx is not None and len(node.args) > idx:
                arg = node.args[idx]
                if _is_string_literal(arg):
                    self._issue(
                        node,
                        "file I/O with a literal path inside a decorated function "
                        "(must go through the injected `log_path`)",
                    )

        self.generic_visit(node)

    def log_path_used_for(self) -> set[str]:
        return set(self._log_path_used)


def load_code_cells() -> list[tuple[int, str]]:
    with NOTEBOOK_PATH.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    return [
        (i, "".join(cell["source"]))
        for i, cell in enumerate(nb["cells"])
        if cell.get("cell_type") == "code"
    ]


def main() -> int:
    cells = load_code_cells()

    issues: list[str] = []
    syntax_errors: list[str] = []
    decorated: list[tuple[str, bool, bool]] = []
    import_sources = 0
    inline_fallbacks: list[str] = []
    read_log_owners: set[str] = set()
    get_log_owners: set[str] = set()
    log_path_used: set[str] = set()

    for idx, src in cells:
        try:
            compile(src, f"<cell_{idx}>", "exec")
            tree = ast.parse(src, filename=f"<cell_{idx}>")
        except SyntaxError as exc:
            syntax_errors.append(
                f"cell {idx}: SyntaxError: {exc.msg} (line {exc.lineno})"
            )
            continue

        checker = CellChecker(idx)
        checker.visit(tree)

        issues.extend(checker.issues)
        decorated.extend(checker.decorated)
        import_sources += int(checker.import_experiment_log)
        inline_fallbacks.extend(checker.inline_fallback)
        read_log_owners |= checker.read_log_owners
        get_log_owners |= checker.get_log_owners
        log_path_used |= checker.log_path_used_for()

    print("=" * 78)
    print("check_log_writes.py — AST-чекер стратегии логирования")
    print("=" * 78)

    if syntax_errors:
        print("\n[FAIL] Синтаксические ошибки в code-ячейках:")
        for e in syntax_errors:
            print("  -", e)
    else:
        print("\n[OK] Синтаксис всех code-ячеек корректен (compile).")

    if issues:
        print("\n[FAIL] Нарушения стратегии логирования:")
        for e in issues:
            print("  -", e)
    else:
        print("[OK] Нет glob, ручных путей, свободных read_logs/литеральных путей.")

    d_names = [n for n, _, _ in decorated]
    dup = {n for n in d_names if d_names.count(n) > 1}
    if sorted(d_names) != sorted(DECORATED_FUNCTIONS) or dup:
        print("\n[FAIL] Ожидались ровно 3 @experiment_log:",
              sorted(DECORATED_FUNCTIONS))
        print("       Найдено:", sorted(d_names), "| дубликаты:", sorted(dup))
    else:
        bad_root = [n for n, r, _ in decorated if not r]
        bad_param = [n for n, _, p in decorated if not p]
        if bad_root or bad_param:
            print("\n[FAIL] Проблемы контракта декораторов:")
            if bad_root:
                print("  - нет root=RESULTS_DIR у:", bad_root)
            if bad_param:
                print("  - нет параметра log_path у:", bad_param)
        else:
            print("[OK] 4 декоратора @experiment_log, у всех root=RESULTS_DIR и log_path найден.")

    missing_use = DECORATED_FUNCTIONS - log_path_used
    if missing_use:
        print("[WARN] log_path не используется в теле:", sorted(missing_use))
    else:
        print("[OK] Запись через log_path используется во всех 4 функциях.")

    if import_sources != 1:
        print(f"[FAIL] `from experiment_log import experiment_log` найден {import_sources} раз (ожидалось 1).")
    else:
        print("[OK] Импорт `from experiment_log import experiment_log` найден.")

    if inline_fallbacks:
        print("[FAIL] Найден inline-fallback experiment_log:", inline_fallbacks)
    else:
        print("[OK] Inline-fallback experiment_log отсутствует.")

    missing_readers = READERS - read_log_owners
    if missing_readers:
        print("[WARN] read_logs() не вызывается для:", sorted(missing_readers))
    else:
        print("[OK] Чтение идёт через .read_logs() у:", sorted(READERS))

    failed = bool(syntax_errors or issues) or (
        sorted(d_names) != sorted(DECORATED_FUNCTIONS)
        or bool(dup)
        or import_sources != 1
        or bool(inline_fallbacks)
    )
    if not syntax_errors:
        bad_root = [n for n, r, _ in decorated if not r]
        bad_param = [n for n, _, p in decorated if not p]
        if bad_root or bad_param:
            failed = True

    print("\n" + "=" * 78)
    if failed:
        print("РЕЗУЛЬТАТ: FAIL")
        return 1
    print("РЕЗУЛЬТАТ: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())