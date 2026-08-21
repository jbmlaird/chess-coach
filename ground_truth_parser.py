import re
from dataclasses import dataclass
from enum import Enum, auto

VERDICT_RE = re.compile(r"VERDICT\s*:[ \t]*(.+)", re.IGNORECASE)


class Outcome(Enum):
    BEST = auto()
    BLUNDER = auto()
    PARSE_ERROR = auto()


@dataclass
class ParsedGroundTruth:
    outcome: Outcome


def extract_verdict(completion: str) -> str | None:
    cleaned = completion.replace("*", "").replace("`", "")

    matches = VERDICT_RE.findall(cleaned)
    if not matches:
        return None

    words = matches[-1].split()
    if not words:
        return None

    return words[0].rstrip(".,;:!?") or None


def parse_ground_truth(completion: str) -> ParsedGroundTruth:
    verdict = extract_verdict(completion)

    if not verdict:
        return ParsedGroundTruth(outcome=Outcome.PARSE_ERROR)

    if verdict.upper() == "BEST":
        return ParsedGroundTruth(outcome=Outcome.BEST)
    if verdict.upper() == "BLUNDER":
        return ParsedGroundTruth(outcome=Outcome.BLUNDER)

    return ParsedGroundTruth(outcome=Outcome.PARSE_ERROR)
