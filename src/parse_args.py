
from dataclasses import dataclass, field

@dataclass
class BasicArguments:
    ln1: str = field(
        default="en",
        metadata={"help": "first language"}
    )
    ln2: str = field(
        default="ru",
        metadata={"help": "second language"}
    )