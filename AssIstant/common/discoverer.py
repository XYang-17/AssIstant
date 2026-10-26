import ast
from sys import modules
from importlib import import_module
from pathlib import Path
from typing import Sequence

class Discover:
    def __init__(self, root: str | Path, files: Sequence[str]):
        self._root = root if isinstance(root, Path) else Path(root)
        self._files = tuple(file if file.endswith(".py") else f"{file}.py" for file in files)

    def discover(self, suffix: str):
        res = set()
        for file in self._root.rglob('*.py'):
            if not file.name == "agents.py" and not file.name.endswith(self._files):
                continue
            # if not file.name.endswith(self._pattern)
            try:
                source = file.read_text(encoding='utf-8')
                tree = ast.parse(source, filename=str(file))
            except (SyntaxError, UnicodeError) as e:
                print(f"warning: {e}")
                continue

            nodes = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name.endswith(suffix)
            ]
            if not nodes:
                continue

            module = self._load_module(file)
            for node in nodes:
                if module is not None:
                    cls = getattr(module, node.name, None)
                #     if cls is None:
                #         print(f"warning: ")
                    if cls is not None:
                        res.add(cls)
        return res


    def _load_module(self, file: Path):
        """按相对路径推导模块名并导入；失败时回退到 spec_from_file_location。"""
        # 方案 A：把 root 当作包根，用点分模块名导入（支持相对 import）
        try:
            rel = file.relative_to(self._root).with_suffix("")
            module_name = ".".join(rel.parts)
            if module_name in modules:       # 已导入过则复用
                return modules[module_name]
            return import_module(module_name)
        except Exception as e:
            print(f"[导入失败-相对路径] {file}: {e}")

        # 方案 B：直接按文件加载（只适用于模块内没有相对导入的情况）
        try:
            import importlib.util
            mod_name = f"_dyn_{file.stem}_{abs(hash(str(file)))}"
            spec = importlib.util.spec_from_file_location(mod_name, file)
            module = importlib.util.module_from_spec(spec)
            modules[mod_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"[导入失败-文件加载] {file}: {e}")
            return None