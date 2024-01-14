import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Paths:
    root: Path = Path(__file__).parent
    data: Path = root / "data"
    book: Path = (
            data
            / "Marcus_Aurelius_Antoninus_-_His_Meditations_concerning_himselfe/index.html"
    )
    mrau_weblive_project1: Path = (
            data
            / "wl_mrproject/mention_1305_2024-01-11-174801.csv"
    )
    mrau_weblive_project1_1k_mentions: Path = (
            data
            / "wl_mrproject/1k_mention_1305_2024-01-11-174801.csv"
    )

    mrau_weblive_project1_5k_mentions: Path = (
            data
            / "wl_mrproject/5k_mention_1305_2024-01-11-174801.csv"
    )


openai_api_key = os.getenv("OPENAI_API_KEY")
