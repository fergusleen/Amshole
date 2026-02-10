#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
CONTENT_PATH = ROOT / 'content' / 'amshole_content.json'
MAP_PATH = ROOT / 'content' / 'frame_map.json'
OUT_DIR = ROOT / 'dist' / 'frames'

MAX_ROWS = 23
LINE_WIDTH = 38
HEADER_KEYS = {'1a', '100a', '101a', '110a', '120a', '130a', '140a', '150a', '160a', '170a', '180a', '199a'}

TOKEN_RE = re.compile(r"\[(?:R|G|Y|B|M|C|W|F|S|N|D|-|n|r|g|y|b|m|c|w|h\.|m\.|l\.|h-|m-|l-|_\+|_- )\]")
ALPHA_RE = re.compile(r"\[([rgbymcw])\[(.*?)\]\]")

# Fix accidental whitespace variant in regex by compiling the strict one used everywhere.
TOKEN_RE = re.compile(r"\[(?:R|G|Y|B|M|C|W|F|S|N|D|-|n|r|g|y|b|m|c|w|h\.|m\.|l\.|h-|m-|l-|_\+|_-)\]")

ZERO_TOKENS = {
    '[R]', '[G]', '[Y]', '[B]', '[M]', '[C]', '[W]', '[F]', '[S]', '[N]', '[D]', '[-]', '[n]',
    '[r]', '[g]', '[y]', '[b]', '[m]', '[c]', '[w]', '[h.]', '[m.]', '[l.]', '[h-]', '[m-]', '[l-]',
    '[_+]', '[_-]'
}


def visible_text(s: str) -> str:
    out = s
    while True:
        nxt = ALPHA_RE.sub(r'\2', out)
        if nxt == out:
            break
        out = nxt
    out = TOKEN_RE.sub('', out)
    return out


def visible_len(s: str) -> int:
    return len(visible_text(s))


def clip_visible(line: str, width: int = LINE_WIDTH) -> str:
    out = []
    vis = 0
    i = 0
    while i < len(line) and vis < width:
        if i + 2 < len(line) and line[i] == '[' and line[i + 1] in 'rgbymcw' and line[i + 2] == '[':
            j = line.find(']]', i + 3)
            if j != -1:
                payload = line[i + 3:j]
                keep = payload[: max(0, width - vis)]
                out.append(f"[{line[i + 1]}[{keep}]]")
                vis += len(keep)
                i = j + 2
                continue
        if line[i] == '[':
            j = line.find(']', i + 1)
            if j != -1:
                tok = line[i:j + 1]
                if tok in ZERO_TOKENS:
                    out.append(tok)
                    i = j + 1
                    continue
        out.append(line[i])
        vis += 1
        i += 1
    return ''.join(out)


def wrap_col(prefix: str, text: str, width: int = LINE_WIDTH):
    text = ' '.join(text.split())
    if not text:
        return [clip_visible(prefix, width)]

    avail = max(8, width - visible_len(prefix))
    words = text.split(' ')
    lines = []
    cur = []
    cur_len = 0

    for word in words:
        wl = len(word)
        if not cur:
            cur = [word]
            cur_len = wl
        elif cur_len + 1 + wl <= avail:
            cur.append(word)
            cur_len += 1 + wl
        else:
            lines.append(clip_visible(prefix + ' '.join(cur), width))
            cur = [word]
            cur_len = wl
    if cur:
        lines.append(clip_visible(prefix + ' '.join(cur), width))
    return lines


def sep_dots(color: str = 'C') -> str:
    return f'[{color.lower()}][m.]'


def sep_solid(color: str = 'C') -> str:
    return f'[{color.lower()}][m-]'


def nav_line() -> str:
    return '[C]0 MAIN   [W]# HELP'


def title_data(title: str) -> str:
    lines = [
        clip_visible(f'[Y]{title}'),
        sep_dots('C'),
    ]
    return '\r\n'.join(lines) + '\r\n'


def content_data(lines) -> str:
    clipped = [clip_visible(line) for line in lines]
    return '\r\n'.join(clipped) + '\r\n'


def count_rows(markup: str) -> int:
    if not markup:
        return 0
    return len(markup.split('\r\n')) - 1


def iter_lines(markup: str):
    parts = markup.split('\r\n')
    if parts and parts[-1] == '':
        parts = parts[:-1]
    return parts


def frame_key(page_no: int, frame_id: str) -> str:
    return f'{page_no}{frame_id}'


def next_frame_id(fid: str) -> str:
    n = ord(fid)
    if n >= ord('z'):
        raise ValueError('No frame-id available after z')
    return chr(n + 1)


def assert_width(markup: str, label: str, key: str):
    for idx, line in enumerate(iter_lines(markup), 1):
        v = visible_len(line)
        if v > LINE_WIDTH:
            raise ValueError(f'{key} {label} line {idx} visible width {v} > {LINE_WIDTH}')


def build_frame(page_no: int, frame_id: str, lines, title: Optional[str] = None, routing=None):
    frame = {
        'pid': {'page-no': page_no, 'frame-id': frame_id},
        'visible': True,
        'frame-type': 'information',
    }
    if title is not None:
        frame['title'] = {'type': 'markup', 'data': title_data(title)}
    frame['content'] = {'type': 'markup', 'data': content_data(lines)}
    if routing is not None:
        frame['routing-table'] = routing

    key = frame_key(page_no, frame_id)
    assert_width(frame['content']['data'], 'content', key)
    if 'title' in frame:
        assert_width(frame['title']['data'], 'title', key)

    total_rows = count_rows(frame['content']['data'])
    if 'title' in frame:
        total_rows += count_rows(frame['title']['data'])
    if total_rows > MAX_ROWS:
        raise ValueError(f'Frame {key} exceeds {MAX_ROWS} rows ({total_rows})')

    return frame


def ensure_content_cap(lines, page_no: int, frame_id: str, title_present: bool):
    cap = MAX_ROWS - (2 if title_present else 0)
    if len(lines) > cap:
        raise ValueError(f'Frame {page_no}{frame_id} content rows {len(lines)} exceeds {cap}')
    return lines


def build_intro(content, fmap):
    intro = content.get('intro_page', {})
    page_no = int(intro.get('page_no', 1))
    frame_id = str(intro.get('frame_id', 'a'))
    title = str(intro.get('title', 'CPC Intro'))

    lines = [clip_visible(l) for l in intro.get('lines', [])]
    if not lines:
        lines = [
            '[Y]WELCOME TO AMSHOLE',
            sep_dots('C'),
            '[W]AMSTRAD CPC INTRO PAGE',
        ]

    lines.append(sep_dots('C'))
    lines.append(nav_line())
    lines = ensure_content_cap(lines, page_no, frame_id, True)

    return build_frame(
        page_no,
        frame_id,
        lines,
        title=title,
        routing=fmap.get('routing_tables', {}).get(frame_key(page_no, frame_id))
    )


def build_main(content, fmap):
    lines = []
    lines += wrap_col('[W]', content['menu']['strapline'])
    lines.append(sep_solid('C'))

    colours = ['Y', 'C', 'G', 'M', 'B', 'Y', 'C', 'G', 'W']
    for idx, opt in enumerate(content['menu']['options']):
        col = colours[idx % len(colours)]
        lines += wrap_col(f'[{col}]', f"{opt['key']} {opt['label']} -> {opt['target']}")

    lines.append(sep_dots('C'))
    lines += wrap_col('[G]', 'Try page 1 for the mosaic CPC intro.')
    lines.append(nav_line())
    lines = ensure_content_cap(lines, 100, 'a', True)

    return build_frame(
        100,
        'a',
        lines,
        title='AMSHOLE Main Index',
        routing=fmap['routing_tables'].get('100a')
    )


def build_section_index(section, fmap):
    lines = []
    lines += wrap_col('[W]', section['index_intro'])
    lines += wrap_col('[G]', section['index_note'])
    lines.append(sep_dots('C'))

    for page in section['pages']:
        lines += wrap_col('[Y]', f"{page['frame_id'].upper()} {page['title']}")

    lines.append(sep_solid('C'))
    lines += wrap_col('[W]', '0 returns to main, # opens help page.')
    lines.append(nav_line())

    page_no = section['page_no']
    lines = ensure_content_cap(lines, page_no, 'a', True)
    return build_frame(
        page_no,
        'a',
        lines,
        title=f"{section['name']} Index",
        routing=fmap['routing_tables'].get(frame_key(page_no, 'a'))
    )


def build_article(section, page):
    lines = []
    lines += wrap_col('[Y]', page['title'])
    lines += wrap_col('[W]', page['intro'])
    lines.append(sep_dots('C'))

    for para in page['paragraphs']:
        lines += wrap_col('[W]', para)

    lines.append(sep_dots('C'))
    for bullet in page['bullets']:
        lines += wrap_col('[G]', f'- {bullet}')

    box = page.get('box')
    if box:
        lines.append(sep_solid('M'))
        lines += wrap_col('[Y]', box['label'])
        lines += wrap_col('[C]', box['text'])

    lines.append(sep_dots('C'))
    lines.append(nav_line())

    lines = ensure_content_cap(lines, section['page_no'], page['frame_id'], False)
    return build_frame(section['page_no'], page['frame_id'], lines)


def paginate_help(help_lines):
    first_cap = MAX_ROWS - 2
    cont_cap = MAX_ROWS
    out = []
    idx = 0
    frame_id = 'a'

    while idx < len(help_lines):
        cap = first_cap if frame_id == 'a' else cont_cap
        remaining = len(help_lines) - idx
        reserve = 2 if remaining > (cap - 1) else 1
        take = cap - reserve
        if take <= 0:
            raise ValueError('Help pagination capacity is too small')

        chunk = list(help_lines[idx:idx + take])
        idx += take

        if idx < len(help_lines):
            next_id = next_frame_id(frame_id)
            chunk.append(clip_visible(f'[Y]CONTINUED ON 199{next_id}'))
        chunk.append(nav_line())

        out.append((frame_id, chunk))
        frame_id = next_frame_id(frame_id)

    return out


def build_help_frames(content, fmap):
    help_data = content['help']
    lines = []
    lines += wrap_col('[W]', help_data['intro'])
    lines.append(sep_dots('C'))

    for para in help_data['paragraphs']:
        lines += wrap_col('[W]', para)

    lines.append(sep_solid('C'))
    lines += wrap_col('[Y]', 'Credits')
    for item in help_data['credits']:
        lines += wrap_col('[G]', f'- {item}')

    lines.append(sep_dots('C'))

    frames = []
    for idx, (fid, chunk) in enumerate(paginate_help(lines)):
        title = help_data['title'] if idx == 0 else None
        routing = fmap['routing_tables'].get('199a') if idx == 0 else None
        frames.append(build_frame(199, fid, chunk, title=title, routing=routing))
    return frames


def write_frame(frame):
    pid = frame['pid']
    path = OUT_DIR / f"{pid['page-no']}{pid['frame-id']}.json"
    with path.open('w', encoding='utf-8') as f:
        json.dump(frame, f, indent=2)
        f.write('\n')


def main():
    content = json.loads(CONTENT_PATH.read_text(encoding='utf-8'))
    fmap = json.loads(MAP_PATH.read_text(encoding='utf-8'))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob('*.json'):
        old.unlink()

    frames = []
    frames.append(build_intro(content, fmap))
    frames.append(build_main(content, fmap))

    sec_by_id = {s['id']: s for s in content['sections']}
    for sec_map in fmap['sections']:
        sec = sec_by_id[sec_map['id']]
        frames.append(build_section_index(sec, fmap))
        for page in sec['pages']:
            frames.append(build_article(sec, page))

    frames.extend(build_help_frames(content, fmap))

    for frame in frames:
        key = frame_key(frame['pid']['page-no'], frame['pid']['frame-id'])
        if key in HEADER_KEYS and 'title' not in frame:
            raise ValueError(f'{key} must include title')
        if key not in HEADER_KEYS and 'title' in frame:
            raise ValueError(f'{key} should not include title')
        write_frame(frame)


if __name__ == '__main__':
    main()
