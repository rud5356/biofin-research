from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import read_csv_flexible


def test_read_csv_flexible_detects_cp949_tab_delimiter(tmp_path: Path) -> None:
    path = tmp_path / "budget.csv"
    path.write_text(
        "No.\t세부사업명\t금액\n1\t테스트 사업\t1,000,000\n",
        encoding="cp949",
    )

    frame = read_csv_flexible(path)

    assert list(frame.columns) == ["No.", "세부사업명", "금액"]
    assert frame.loc[0, "세부사업명"] == "테스트 사업"
    assert frame.loc[0, "금액"] == "1,000,000"
