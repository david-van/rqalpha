"""
扫描 xiaoe_articles/ 目录生成文章索引页面 index.html。
用法：python generate_index.py
"""
import os
import re
import json

BASE_DIR = "xiaoe_articles"
OUTPUT_FILE = os.path.join(BASE_DIR, "index.html")

FILENAME_RE = re.compile(r'^(\d{4}-\d{2}-\d{2})_(.+)\.html$')


def scan_articles() -> list[dict]:
    articles = []
    for entry in os.scandir(BASE_DIR):
        if not entry.is_dir():
            continue
        year_dir = entry.name
        for f in os.scandir(entry.path):
            if not f.is_file() or not f.name.endswith(".html"):
                continue
            m = FILENAME_RE.match(f.name)
            if not m:
                continue
            date_str, title = m.group(1), m.group(2)
            articles.append({
                "year": year_dir,
                "date": date_str,
                "title": title,
                "path": f"{year_dir}/{f.name}",
            })
    articles.sort(key=lambda a: a["date"], reverse=True)
    return articles


def group_by_year(articles: list[dict]) -> list[dict]:
    years = []
    current_year = None
    for a in articles:
        if a["year"] != current_year:
            current_year = a["year"]
            years.append({"year": current_year, "articles": []})
        years[-1]["articles"].append(a)
    return years


def generate():
    articles = scan_articles()
    if not articles:
        print("未找到任何文章，请先运行 download_xiaoe.py 下载文章。")
        return

    years = group_by_year(articles)
    articles_json = json.dumps(articles, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>投资日志</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ display: flex; height: 100vh; font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f5f5f5; }}
a {{ color: #333; text-decoration: none; }}

/* 侧边栏 */
#sidebar {{
  width: var(--sidebar-width, 300px); min-width: 200px; height: 100vh; background: #fff;
  display: flex; flex-direction: column; overflow: hidden;
}}
/* 拖拽手柄 */
#resize-handle {{
  width: 5px; cursor: col-resize; background: transparent;
  transition: background .2s; flex-shrink: 0;
}}
#resize-handle:hover, #resize-handle.dragging {{
  background: #4a90d9;
}}
#search-box {{
  padding: 12px 16px; border-bottom: 1px solid #eee;
}}
#search-box input {{
  width: 100%; padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px;
  font-size: 14px; outline: none;
}}
#search-box input:focus {{ border-color: #4a90d9; }}
#article-list {{
  flex: 1; overflow-y: auto; padding: 8px 0;
}}
.year-group {{ margin-bottom: 4px; }}
.year-header {{
  display: flex; align-items: center; padding: 10px 16px; cursor: pointer;
  font-size: 15px; font-weight: bold; color: #555; user-select: none;
  border-bottom: 1px solid #f0f0f0;
}}
.year-header:hover {{ background: #f9f9f9; }}
.year-header .arrow {{
  display: inline-block; width: 16px; transition: transform .2s;
  font-size: 12px; color: #999;
}}
.year-header .arrow.open {{ transform: rotate(90deg); }}
.year-count {{ font-weight: normal; font-size: 12px; color: #999; margin-left: 6px; }}
.article-items {{ overflow: hidden; }}
.article-items.collapsed {{ display: none; }}
.article-item {{
  display: block; padding: 8px 16px 8px 32px; font-size: 14px; color: #444;
  cursor: pointer; border-left: 3px solid transparent;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.article-item:hover {{ background: #eef4fb; }}
.article-item.active {{ background: #dce9f7; border-left-color: #4a90d9; color: #1a6fc4; font-weight: 500; }}
.article-item .date {{ font-size: 11px; color: #999; margin-right: 6px; }}
.no-result {{ padding: 20px; text-align: center; color: #999; font-size: 14px; }}

/* 内容区 */
#content {{ flex: 1; display: flex; flex-direction: column; }}
#content iframe {{ flex: 1; border: none; background: #fff; }}
#welcome {{
  display: flex; align-items: center; justify-content: center; height: 100%;
  color: #bbb; font-size: 18px;
}}

</style>
</head>
<body>

<div id="sidebar">
  <div id="search-box">
    <input type="text" id="search" placeholder="搜索文章标题...">
  </div>
  <div id="article-list"></div>
</div>

<div id="resize-handle"></div>

<div id="content">
  <div id="welcome">选择左侧文章开始阅读</div>
  <iframe id="frame" style="display:none"></iframe>
</div>

<script>
const DATA = {articles_json};
const FRAME = document.getElementById('frame');
const WELCOME = document.getElementById('welcome');
const LIST = document.getElementById('article-list');
const SEARCH = document.getElementById('search');

// group by year
const groups = [];
let currentYear = null;
for (const a of DATA) {{
  if (a.year !== currentYear) {{
    currentYear = a.year;
    groups.push({{ year: currentYear, articles: [] }});
  }}
  groups[groups.length - 1].articles.push(a);
}}

function showArticle(path, el) {{
  document.querySelectorAll('.article-item').forEach(i => i.classList.remove('active'));
  el.classList.add('active');
  WELCOME.style.display = 'none';
  FRAME.style.display = '';
  FRAME.src = path;
  localStorage.setItem('lastArticle', path);
}}

function render(filter) {{
  const q = (filter || '').toLowerCase();
  LIST.innerHTML = '';
  let visible = 0;
  for (const g of groups) {{
    const visibleArticles = g.articles.filter(a => a.title.toLowerCase().includes(q));
    if (visibleArticles.length === 0) continue;
    visible += visibleArticles.length;

    const isLastYear = g.year === groups[0].year;
    const div = document.createElement('div');
    div.className = 'year-group';
    div.innerHTML = `<div class="year-header" data-year="${{g.year}}">
      <span class="arrow${{isLastYear ? ' open' : ''}}">▶</span>
      ${{g.year}} 年<span class="year-count">(${{visibleArticles.length}}篇)</span>
    </div>
    <div class="article-items${{isLastYear ? '' : ' collapsed'}}"></div>`;

    const items = div.querySelector('.article-items');
    for (const a of visibleArticles) {{
      const item = document.createElement('div');
      item.className = 'article-item';
      item.innerHTML = `<span class="date">${{a.date.slice(5)}}</span><span>${{a.title}}</span>`;
      item.onclick = () => showArticle(a.path, item);
      items.appendChild(item);
    }}

    div.querySelector('.year-header').onclick = function() {{
      const arrow = this.querySelector('.arrow');
      const items = this.nextElementSibling;
      arrow.classList.toggle('open');
      items.classList.toggle('collapsed');
    }};

    LIST.appendChild(div);
  }}
  if (visible === 0 && q) {{
    LIST.innerHTML = '<div class="no-result">没有匹配的文章</div>';
  }}
}}

SEARCH.oninput = () => render(SEARCH.value);
render();

// 侧边栏拖拽调整宽度
const HANDLE = document.getElementById('resize-handle');
const SIDEBAR = document.getElementById('sidebar');
let savedWidth = localStorage.getItem('sidebarWidth');
if (savedWidth) SIDEBAR.style.setProperty('--sidebar-width', savedWidth + 'px');

HANDLE.onmousedown = (e) => {{
  e.preventDefault();
  HANDLE.classList.add('dragging');
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';

  const onMove = (ev) => {{
    const w = Math.max(200, Math.min(800, ev.clientX));
    SIDEBAR.style.setProperty('--sidebar-width', w + 'px');
  }};
  const onUp = () => {{
    HANDLE.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    FRAME.style.pointerEvents = '';
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    localStorage.setItem('sidebarWidth', parseInt(SIDEBAR.style.getPropertyValue('--sidebar-width')));
  }};
  FRAME.style.pointerEvents = 'none';
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}};

// restore last article
const last = localStorage.getItem('lastArticle');
if (last) {{
  FRAME.src = last;
  FRAME.style.display = '';
  WELCOME.style.display = 'none';
}}
</script>
</body>
</html>"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成索引页面: {OUTPUT_FILE}（共 {len(articles)} 篇文章）")


if __name__ == "__main__":
    generate()
