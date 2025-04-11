import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents in the system.

    An agent is responsible for handling a specific type of query or task.
    New agents can be created by inheriting from this class and implementing
    the required methods.
    """

    def __init__(self, name: str, verbose: bool = False):
        """
        Initialize a base agent.

        Args:
            name (str): The name of the agent.
            verbose (bool): Whether to enable verbose logging.
        """
        self.name = name
        self.verbose = verbose

    def log(self, message: str, level: str = "info"):
        """Log a message if verbose is enabled."""
        if not self.verbose:
            return

        if level == "info":
            logger.info(f"[{self.name}] {message}")
        elif level == "error":
            logger.error(f"[{self.name}] {message}")
        elif level == "warning":
            logger.warning(f"[{self.name}] {message}")
        elif level == "debug":
            logger.debug(f"[{self.name}] {message}")

    @abstractmethod
    def run(self, state: Dict, **kwargs) -> Dict[str, Any]:
        """
        Process the given query and return a result.
        """
        pass

    @abstractmethod
    async def arun(self, state: Dict, **kwargs) -> Dict[str, Any]:
        """
        Asynchronously process the given query and return a result.
        """
        pass
