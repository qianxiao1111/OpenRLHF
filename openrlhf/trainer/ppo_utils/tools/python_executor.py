import io
import re
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
from typing import Any, Dict, Optional
from timeout_decorator import timeout
from contextlib import redirect_stdout
from base_tool import BaseTool

class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r"(\s|^)?input\(", code_piece):
            # regex.search(r'(\s|^)?os.', code_piece):
            raise RuntimeError()
        exec(code_piece, self._global_vars)
    
    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars["answer"]

class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        "datetime": datetime.datetime,
        "timedelta": dateutil.relativedelta.relativedelta,
        "relativedelta": dateutil.relativedelta.relativedelta,
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {"dict": CustomDict}

class PythonExecutor(BaseTool):
    
    def __init__(
        self, 
        runtime: Optional[Any] = None,
        timeout_length: Optional[int] = 20
    ):
        super().__init__(name="PythonExecutor", description="Execute Python code in a sandboxed environment.")
        self.runtime = runtime if runtime else GenericRuntime()
        self.timeout_length = timeout_length
    
    def execute(self, input_context, *args, **kwargs):
        code_input = self.parse_input(input_context)
        if not code_input:
            return "The input code is empty or invalid."
        try:
            program_io = io.StringIO()
            with redirect_stdout(program_io):
                timeout(self.timeout_length)(self.runtime.exec_code)(code_input)
            program_io.seek(0)
            result = program_io.read()
            report = "Done"
            str(result)
            pickle.dumps(result)  # serialization check
        except:
            result = ""
            report = traceback.format_exc().split("\n")[-2]
        return result, report

    def parse_input(self, raw_input: str):

        def extract_python_code(text, pattern_model="loose"):
            """
            Extracts Markdown Python code blocks from a given text. Supports one or multiple code blocks.
            Code not enclosed in Markdown format cannot be extracted.

            Cases:
            1. No matches found: returns an empty list []
            2. One match found: returns a list
            3. Multiple matches found: returns a list
            4. Matches found, but code is unusable (empty or not Python code): returns None to indicate parsing failure

            pattern: 'loose' or 'strict'
            """
            original_text = copy.deepcopy(text)
            python_code_list = []
            if pattern_model == "loose":
                pattern = r"``+python\s*\n?(.*?)``+"
            elif pattern_model == "strict":
                pattern = r"```python\s*\n?(.*?)```"
            else:
                raise ValueError("pattern_model not allow")

            matches = re.findall(pattern, original_text, re.DOTALL)  # 检查是否有匹配
            if matches:
                for match in matches:
                    code_block = match.strip()
                    python_code_list.append(code_block)

            count_n = original_text.count("``python")
            if not count_n == len(python_code_list):
                return "The pyhton code format is incorrect."
            else:
                return "\n".join(python_code_list).strip()
            
        tool_input = extract_python_code(raw_input)

        return tool_input
