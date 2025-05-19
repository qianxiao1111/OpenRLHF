from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseTool(ABC):
    name: str = ""
    description: str = ""
    args: Optional[dict | str] = None  # 可以是字典或者字符串

    @abstractmethod
    def execute(self, input_context, *args, **kwargs):
        """
        Execute the tool's main functionality.

        Args:
            input_context: The required input context for the tool.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        This method should be implemented by subclasses.
        """
        pass

    def parse_input(self, raw_input):
        """
        Parse the raw input into a suitable format for the tool.

        Args:
            raw_input: The raw input data to be parsed.

        Returns:
            Parsed input in the required format.

        This method should be implemented by subclasses.
        """
        return raw_input