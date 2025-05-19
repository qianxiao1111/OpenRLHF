import re
import sqlite3
from typing import Any, List, Tuple
import threading
import time

from .base_tool import BaseTool  # 假设BaseTool在同目录下

class SQLiteExecutor(BaseTool):
    def __init__(self, db_file_path: str, timeout_length: float = 5.0):
        self.db_file_path = db_file_path
        self.timeout_length = timeout_length

    def parse_input(self, input_text: str) -> str:
        pattern = r"```sql\s*(.*?)\s*```"
        match = re.search(pattern, input_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            raise ""

    def _execute_sql(self, sql: str, result_container: dict):
        conn = sqlite3.connect(self.db_file_path)
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            if cursor.description:  # Check if the query returns results
                result_container['result'] = cursor.fetchall()
            else:
                result_container['result'] = []
            conn.commit()
        except Exception as e:
            result_container['error'] = str(e)
        finally:
            conn.close()

    def execute(self, input_context: str) -> List[Tuple[Any, ...]]:
        sql = self.parse_input(input_context)
        result_container = {}
        if sql:
            thread = threading.Thread(target=self._execute_sql, args=(sql, result_container))
            thread.start()
            thread.join(self.timeout_length)
            if thread.is_alive():
                return [("Timeout: SQL execution exceeded {} seconds.".format(self.timeout_length),)]
            if 'error' in result_container:
                return [("Error: " + result_container['error'],)]
            return result_container.get('result', [])
        else:
            return None