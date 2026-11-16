"""
view_visualizations.py

Tạo một trang HTML (gallery) để xem TẤT CẢ ảnh trong
reports/05_pseudo_gt/visualization/ và mở bằng trình duyệt.

Tính năng:
  - Mỗi ảnh hiển thị kèm: cell_type, defect_probability, mask area 3 model
  - Lọc theo nhóm (normal / borderline / defect) và cell type (mono / poly)
  - Sắp xếp theo tên, defect_probability, hoặc mask area của từng model
  - Click vào ảnh để mở bản gốc ở tab mới
  - Dùng đường dẫn tương đối nên mở bằng file:// vẫn chạy

Cách chạy (từ project root):
  python scripts/view_visualizations.py
  python scripts/view_visualizations.py --no-open     # chỉ tạo file, không mở

Output:
  reports/05_pseudo_gt/visualization/gallery.html

Không sửa, không xoá file nào có sẵn.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIS_DIR      = PROJECT_ROOT / "reports" / "05_pseudo_gt" / "visualization"
SUMMARY_CSV  = PROJECT_ROOT / "reports" / "05_pseudo_gt" / "summary.csv"
OUT_HTML     = VIS_DIR / "gallery.html"

PROB_ZERO_THRESHOLD = 0.05
PROB_ONE_THRESHOLD  = 0.95


def prob_group(p: float) -> str:
    if p <= PROB_ZERO_THRESHOLD:
        return "normal"
    if p >= PROB_ONE_THRESHOLD:
        return "defect"
    return "borderline"


def build_records() -> list[dict]:
    if not VIS_DIR.exists():
        sys.exit(f"Không thấy thư mục: {VIS_DIR}")

    files = sorted(p.name for p in VIS_DIR.glob("*.png"))
    if not files:
        sys.exit(f"Thư mục không có ảnh .png: {VIS_DIR}")

    meta = {}
    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        df["stem"] = df["image_id"].map(lambda s: Path(s).stem)
        meta = df.set_index("stem").to_dict("index")
    else:
        print(f"[cảnh báo] Không có {SUMMARY_CSV}, sẽ chỉ hiển thị tên file.")

    records = []
    for fname in files:
        stem = Path(fname).stem
        row = meta.get(stem)
        if row is None:
            records.append({
                "file": fname, "id": stem, "cell": "?", "prob": None,
                "group": "unknown",
                "pc": None, "ae": None, "vae": None,
            })
            continue
        prob = float(row["defect_probability"])
        records.append({
            "file":  fname,
            "id":    stem,
            "cell":  str(row["cell_type"]),
            "prob":  round(prob, 3),
            "group": prob_group(prob),
            "pc":    round(100 * float(row["patchcore_mask_area"]), 2),
            "ae":    round(100 * float(row["ae_mask_area"]), 2),
            "vae":   round(100 * float(row["vae_mask_area"]), 2),
        })
    return records


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>ELPV pseudo-GT gallery</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f4f4f2; color: #222; }
  header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd;
           padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; z-index: 5; }
  header label { font-size: 13px; color: #555; }
  select, button { font-size: 13px; padding: 4px 8px; }
  #count { font-size: 13px; color: #555; margin-left: auto; }
  main { padding: 16px; display: flex; flex-direction: column; gap: 16px; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
  .meta { font-size: 13px; margin-bottom: 6px; display: flex; flex-wrap: wrap; gap: 6px 14px; }
  .tag { padding: 1px 8px; border-radius: 10px; font-size: 12px; }
  .normal { background: #e1f5ee; color: #085041; }
  .borderline { background: #faeeda; color: #633806; }
  .defect { background: #fcebeb; color: #791f1f; }
  .unknown { background: #eee; color: #444; }
  .card img { width: 100%; height: auto; display: block; cursor: zoom-in; }
  .nums { color: #555; }
</style>
</head>
<body>
<header>
  <label>Nhóm
    <select id="fGroup">
      <option value="all">Tất cả</option>
      <option value="normal">normal (prob &le; 0.05)</option>
      <option value="borderline">borderline</option>
      <option value="defect">defect (prob &ge; 0.95)</option>
    </select>
  </label>
  <label>Cell
    <select id="fCell">
      <option value="all">Tất cả</option>
      <option value="mono">mono</option>
      <option value="poly">poly</option>
    </select>
  </label>
  <label>Sắp xếp
    <select id="fSort">
      <option value="id">Tên file</option>
      <option value="prob">defect_probability (cao &rarr; thấp)</option>
      <option value="pc">PatchCore mask area (cao &rarr; thấp)</option>
      <option value="ae">AE mask area (cao &rarr; thấp)</option>
      <option value="vae">VAE mask area (cao &rarr; thấp)</option>
    </select>
  </label>
  <button id="reset">Đặt lại</button>
  <span id="count"></span>
</header>
<main id="grid"></main>

<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);

function render() {
  const g = $("fGroup").value, c = $("fCell").value, s = $("fSort").value;
  let rows = DATA.filter(r => (g === "all" || r.group === g) && (c === "all" || r.cell === c));
  if (s === "id") rows.sort((a, b) => a.id.localeCompare(b.id));
  else rows.sort((a, b) => (b[s] ?? -1) - (a[s] ?? -1));

  $("count").textContent = rows.length + " / " + DATA.length + " ảnh";
  $("grid").innerHTML = rows.map(r => {
    const probTxt = r.prob === null ? "?" : r.prob.toFixed(3);
    const nums = r.pc === null ? "" :
      `<span class="nums">mask: PatchCore ${r.pc}% | AE ${r.ae}% | VAE ${r.vae}%</span>`;
    return `<div class="card">
      <div class="meta">
        <strong>${r.id}</strong>
        <span class="tag ${r.group}">${r.group}</span>
        <span>cell: ${r.cell}</span>
        <span>prob: ${probTxt}</span>
        ${nums}
      </div>
      <a href="${r.file}" target="_blank"><img loading="lazy" src="${r.file}" alt="${r.id}"></a>
    </div>`;
  }).join("");
}

["fGroup", "fCell", "fSort"].forEach(id => $(id).addEventListener("change", render));
$("reset").addEventListener("click", () => {
  $("fGroup").value = "all"; $("fCell").value = "all"; $("fSort").value = "id"; render();
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true",
                        help="Chỉ tạo gallery.html, không mở trình duyệt")
    args = parser.parse_args()

    records = build_records()
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(records, ensure_ascii=False))
    OUT_HTML.write_text(html, encoding="utf-8")

    n_known = sum(r["group"] != "unknown" for r in records)
    print(f"Đã tạo gallery với {len(records)} ảnh ({n_known} ảnh có metadata):")
    print(f"  {OUT_HTML}")

    if not args.no_open:
        webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()