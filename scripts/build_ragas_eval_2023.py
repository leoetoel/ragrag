"""
Build a 20-item RAGAS retrieval eval dataset from the 2023 annual report MD.
"""

import json
from pathlib import Path


def find_report_md(data_dir: Path) -> Path:
    candidates = [p for p in data_dir.glob("*.md") if "2023" in p.name]
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit(f"Expected 1 report md, found: {len(candidates)}")


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    report_md = find_report_md(data_dir)
    lines = report_md.read_text(encoding="utf-8", errors="ignore").splitlines()

    def line(n: int) -> str:
        return lines[n - 1].strip()

    items = [
        {
            "question": "2023年度公司每10股派发现金红利是多少元？",
            "ground_truth": "308.76元（含税）",
            "contexts": [line(28)],
            "source": str(report_md),
            "line": 28,
        },
        {
            "question": "截至2023年12月31日，公司总股本是多少万股？",
            "ground_truth": "125,619.78万股",
            "contexts": [line(28)],
            "source": str(report_md),
            "line": 28,
        },
        {
            "question": "2023年度拟派发现金红利总额是多少元？",
            "ground_truth": "38,786,363,272.80元（含税）",
            "contexts": [line(28)],
            "source": str(report_md),
            "line": 28,
        },
        {
            "question": "2023年基本每股收益是多少元/股？",
            "ground_truth": "59.49元/股",
            "contexts": [line(130)],
            "source": str(report_md),
            "line": 130,
        },
        {
            "question": "2023年稀释每股收益是多少元/股？",
            "ground_truth": "59.49元/股",
            "contexts": [line(131)],
            "source": str(report_md),
            "line": 131,
        },
        {
            "question": "2023年经营活动产生的现金流量净额是多少元？",
            "ground_truth": "66,593,247,721.09元",
            "contexts": [line(146)],
            "source": str(report_md),
            "line": 146,
        },
        {
            "question": "2023年末总资产是多少元？",
            "ground_truth": "272,699,660,092.25元",
            "contexts": [line(150)],
            "source": str(report_md),
            "line": 150,
        },
        {
            "question": "2023年营业成本是多少元？",
            "ground_truth": "11,867,273,851.78元",
            "contexts": [line(251)],
            "source": str(report_md),
            "line": 251,
        },
        {
            "question": "2023年投资活动产生的现金流量净额是多少元？",
            "ground_truth": "-9,724,414,015.16元",
            "contexts": [line(257)],
            "source": str(report_md),
            "line": 257,
        },
        {
            "question": "2023年筹资活动产生的现金流量净额是多少元？",
            "ground_truth": "-58,889,101,991.94元",
            "contexts": [line(258)],
            "source": str(report_md),
            "line": 258,
        },
        {
            "question": "公司2023年全年共计派发现金红利是多少亿元？",
            "ground_truth": "565.5亿元",
            "contexts": [line(263)],
            "source": str(report_md),
            "line": 263,
        },
        {
            "question": "2023年现金分红金额占归母净利润的比例是多少？",
            "ground_truth": "75.67%",
            "contexts": [line(263)],
            "source": str(report_md),
            "line": 263,
        },
        {
            "question": "2023年公司营业总收入是多少亿元？",
            "ground_truth": "1,505.60亿元",
            "contexts": [line(267)],
            "source": str(report_md),
            "line": 267,
        },
        {
            "question": "2023年公司营业总收入同比增长多少？",
            "ground_truth": "18.04%",
            "contexts": [line(267)],
            "source": str(report_md),
            "line": 267,
        },
        {
            "question": "2023年归属于上市公司股东的净利润是多少亿元？",
            "ground_truth": "747.34亿元",
            "contexts": [line(267)],
            "source": str(report_md),
            "line": 267,
        },
        {
            "question": "2023年归属于上市公司股东的净利润同比增长多少？",
            "ground_truth": "19.16%",
            "contexts": [line(267)],
            "source": str(report_md),
            "line": 267,
        },
        {
            "question": "2023年本期费用化研发投入是多少元？",
            "ground_truth": "477,957,725.95元",
            "contexts": [line(370)],
            "source": str(report_md),
            "line": 370,
        },
        {
            "question": "2023年本期资本化研发投入是多少元？",
            "ground_truth": "143,549,809.92元",
            "contexts": [line(372)],
            "source": str(report_md),
            "line": 372,
        },
        {
            "question": "2023年研发投入合计是多少元？",
            "ground_truth": "621,507,535.87元",
            "contexts": [line(373)],
            "source": str(report_md),
            "line": 373,
        },
        {
            "question": "2023年研发投入总额占营业收入比例是多少？",
            "ground_truth": "0.42%",
            "contexts": [line(374)],
            "source": str(report_md),
            "line": 374,
        },
    ]

    out_path = data_dir / "ragas_eval_2023.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(out_path)
    print(f"items: {len(items)}")


if __name__ == "__main__":
    main()
