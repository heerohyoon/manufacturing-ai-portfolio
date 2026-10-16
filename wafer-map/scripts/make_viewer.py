"""학습용 웨이퍼 맵을 패턴별로 넘겨볼 수 있는 HTML 뷰어를 만든다. 평가 데이터는 쓰지 않는다.

결과: data/processed/viewer.html (git 미포함). 브라우저로 열면 된다.
실행 (wafer-map/ 폴더에서): python scripts/make_viewer.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_train  # noqa: E402

PER_CLASS = 40
OUT_PATH = ROOT / "data" / "processed" / "viewer.html"

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>웨이퍼 맵 뷰어</title>
<style>
  :root { --bg:#f7f7f5; --card:#fff; --text:#222; --muted:#666; --line:#ddd; --none:#ececec; --pass:#9fc5a8; --fail:#c0392b; --accent:#2f5d8a; }
  @media (prefers-color-scheme: dark) { :root { --bg:#1b1b1d; --card:#252528; --text:#eee; --muted:#aaa; --line:#3a3a3e; --none:#2f2f33; --pass:#4f7a5a; --fail:#e0574a; --accent:#7fb0e0; } }
  * { box-sizing:border-box; }
  body { margin:0; padding:16px; background:var(--bg); color:var(--text); font:14px/1.5 -apple-system, "Apple SD Gothic Neo", sans-serif; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--muted); margin:0 0 12px; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:12px; color:var(--muted); }
  .legend i { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:4px; }
  .tabs { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px; }
  .tabs button { border:1px solid var(--line); background:var(--card); color:var(--text); border-radius:16px; padding:5px 11px; cursor:pointer; font:inherit; }
  .tabs button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  .desc { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:12px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(112px, 1fr)); gap:10px; }
  .cell { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:6px; cursor:pointer; text-align:center; }
  .cell canvas { width:100%; aspect-ratio:1; image-rendering:pixelated; }
  .cell small { color:var(--muted); display:block; }
  dialog { border:1px solid var(--line); border-radius:10px; background:var(--card); color:var(--text); max-width:min(920px, 96vw); width:96vw; padding:16px; }
  dialog::backdrop { background:rgba(0,0,0,.45); }
  .detail { display:flex; gap:16px; flex-wrap:wrap; }
  .detail canvas { width:min(320px, 80vw); aspect-ratio:1; image-rendering:pixelated; border:1px solid var(--line); }
  .nums { font:11px/1.15 ui-monospace, Menlo, monospace; overflow:auto; max-height:60vh; white-space:pre; flex:1; min-width:240px; }
  .nums b { color:var(--fail); }
  .nums span { color:var(--muted); opacity:.45; }
  .close { float:right; border:1px solid var(--line); background:transparent; color:var(--text); border-radius:6px; padding:3px 9px; cursor:pointer; }
</style></head><body>
<h1>웨이퍼 맵 뷰어 (학습 데이터)</h1>
<p class="sub">패턴별로 무작위 __PER__개씩 보여준다. 그림을 누르면 크게 보고 칸 값(0/1/2)을 확인할 수 있다. 평가 데이터는 포함하지 않았다.</p>
<div class="legend"><span><i style="background:var(--none)"></i>0 = 칩 없음 (웨이퍼 바깥)</span><span><i style="background:var(--pass)"></i>1 = 정상 칩</span><span><i style="background:var(--fail)"></i>2 = 불량 칩</span></div>
<div class="tabs" id="tabs"></div>
<div class="desc" id="desc"></div>
<div class="grid" id="grid"></div>
<dialog id="dlg"><button class="close" onclick="dlg.close()">닫기</button><div id="info"></div><div class="detail"><canvas id="big"></canvas><div class="nums" id="nums"></div></div></dialog>
<script>
const DATA = __DATA__;
const DESC = {
  "Center": "웨이퍼 가운데에 불량이 몰린 패턴.",
  "Donut": "가운데는 비교적 괜찮고, 그 바깥 고리 모양으로 불량이 생긴 패턴.",
  "Edge-Loc": "가장자리의 한 부분에 불량이 몰린 패턴.",
  "Edge-Ring": "가장자리를 따라 고리 모양으로 불량이 생긴 패턴.",
  "Loc": "가장자리가 아닌 안쪽 한 부분에 불량이 몰린 패턴.",
  "Near-full": "거의 모든 칩이 불량인 웨이퍼.",
  "Random": "불량이 특정 모양 없이 넓게 흩어진 패턴.",
  "Scratch": "긁힌 자국처럼 선을 따라 불량이 생긴 패턴.",
  "none": "특정 패턴이 없는 웨이퍼. 불량 칩이 몇 개 흩어져 있을 수는 있다."
};
const css = getComputedStyle(document.documentElement);
const COLORS = () => [css.getPropertyValue("--none"), css.getPropertyValue("--pass"), css.getPropertyValue("--fail")].map(s => s.trim());
function draw(canvas, w) {
  const [h, wd] = [w.map.length, w.map[0].length], n = Math.max(h, wd);
  canvas.width = n; canvas.height = n;
  const ctx = canvas.getContext("2d"), c = COLORS();
  ctx.fillStyle = c[0]; ctx.fillRect(0, 0, n, n);
  const oy = Math.floor((n - h) / 2), ox = Math.floor((n - wd) / 2);
  for (let i = 0; i < h; i++) for (let j = 0; j < wd; j++) { ctx.fillStyle = c[w.map[i][j]]; ctx.fillRect(ox + j, oy + i, 1, 1); }
}
function failRate(w) { let d = 0, f = 0; for (const r of w.map) for (const v of r) { if (v > 0) d++; if (v === 2) f++; } return d ? f / d : 0; }
function show(cls) {
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("on", b.dataset.c === cls));
  document.getElementById("desc").innerHTML = `<b>${cls}</b> — ${DESC[cls]} 학습 데이터에 <b>${DATA.counts[cls].toLocaleString()}</b>개.`;
  const grid = document.getElementById("grid"); grid.innerHTML = "";
  DATA.samples[cls].forEach(w => {
    const el = document.createElement("div"); el.className = "cell";
    const cv = document.createElement("canvas"); draw(cv, w);
    el.append(cv); el.insertAdjacentHTML("beforeend", `<small>${w.map.length}×${w.map[0].length} · 불량 ${(failRate(w) * 100).toFixed(0)}%</small>`);
    el.onclick = () => open_(w); grid.append(el);
  });
}
function open_(w) {
  draw(document.getElementById("big"), w);
  const dies = w.map.flat().filter(v => v > 0).length, fails = w.map.flat().filter(v => v === 2).length;
  document.getElementById("info").innerHTML = `<b>${w.label}</b> · 로트 ${w.lot} · ${w.wafer_index}번째 웨이퍼 · 크기 ${w.map.length}×${w.map[0].length} · 칩 ${dies}개 중 불량 ${fails}개 (${(fails / dies * 100).toFixed(1)}%)`;
  const nums = document.getElementById("nums");
  if (w.map.length * w.map[0].length > 4000) nums.textContent = "맵이 커서 칸 값 표시는 생략했다.";
  else nums.innerHTML = w.map.map(r => r.map(v => v === 2 ? "<b>2</b>" : v === 0 ? "<span>0</span>" : "1").join(" ")).join("\\n");
  dlg.showModal();
}
const tabs = document.getElementById("tabs");
DATA.order.forEach(c => { const b = document.createElement("button"); b.dataset.c = c; b.textContent = `${c} (${DATA.counts[c].toLocaleString()})`; b.onclick = () => show(c); tabs.append(b); });
show(DATA.order[0]);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => show(document.querySelector("#tabs .on").dataset.c));
</script></body></html>
"""


def main() -> None:
    df = load_train()
    rng = np.random.default_rng(0)
    counts = df["label"].value_counts()
    order = [c for c in CLASSES if c != "none"] + ["none"]
    samples = {}
    for cls in order:
        rows = df[df["label"] == cls]
        pick = rows.iloc[rng.choice(len(rows), size=min(PER_CLASS, len(rows)), replace=False)]
        samples[cls] = [{"map": r.wafer_map.astype(int).tolist(), "label": cls, "lot": r.lot,
                         "wafer_index": int(r.wafer_index)} for r in pick.itertuples()]
    data = {"order": order, "counts": {c: int(counts.get(c, 0)) for c in order}, "samples": samples}
    html = PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__PER__", str(PER_CLASS))
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"[saved] {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
