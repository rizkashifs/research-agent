from dataclasses import dataclass, field


@dataclass
class Step:
    thought: str
    action: str           # tool name or "final_answer"
    action_input: dict
    observation: str
    iteration: int


@dataclass
class AgentState:
    task: str
    steps: list[Step] = field(default_factory=list)
    iteration: int = 0
    is_complete: bool = False
    final_answer: str = ""
    tool_calls_made: list[str] = field(default_factory=list)  # for eval
