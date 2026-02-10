#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = ROOT / 'dist' / 'frames'
REQUIRED = {'1a', '100a', '199a'}
ROUTED_ONLY = {'1a', '100a', '101a', '110a', '120a', '130a', '140a', '150a', '160a', '170a', '180a', '199a'}
HEADER_KEYS = {'1a', '100a', '101a', '110a', '120a', '130a', '140a', '150a', '160a', '170a', '180a', '199a'}
NAV_TEXT = '[C]0 MAIN   [W]# HELP'
MAX_ROWS = 23
LINE_WIDTH = 38

TOKEN_RE = re.compile(r"\[(?:R|G|Y|B|M|C|W|F|S|N|D|-|n|r|g|y|b|m|c|w|h\.|m\.|l\.|h-|m-|l-|_\+|_-)\]")
ALPHA_RE = re.compile(r"\[([rgbymcw])\[(.*?)\]\]")


def plus_outside_strings(text):
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '+':
                return True
    return False


def visible_text(s):
    out = s
    while True:
        nxt = ALPHA_RE.sub(r'\2', out)
        if nxt == out:
            break
        out = nxt
    return TOKEN_RE.sub('', out)


def visible_len(s):
    return len(visible_text(s))


def row_count(markup):
    if not isinstance(markup, str):
        return 0
    return len(markup.split('\r\n')) - 1


def lines(markup):
    if not isinstance(markup, str):
        return []
    parts = markup.split('\r\n')
    if parts and parts[-1] == '':
        parts = parts[:-1]
    return parts


def main():
    errors = []
    files = sorted(FRAMES_DIR.glob('*.json'))
    if not files:
        errors.append('No frame files found in dist/frames.')

    frame_keys = set()
    pages_present = set()
    parsed = {}

    for path in files:
        raw = path.read_text(encoding='utf-8')
        if plus_outside_strings(raw):
            errors.append(f'{path.name}: contains + outside JSON strings')

        try:
            obj = json.loads(raw)
        except Exception as ex:
            errors.append(f'{path.name}: invalid JSON: {ex}')
            continue

        parsed[path] = obj
        stem = path.stem
        frame_keys.add(stem)

        pid = obj.get('pid')
        if not isinstance(pid, dict):
            errors.append(f'{path.name}: missing pid object')
            continue

        page_no = pid.get('page-no')
        frame_id = pid.get('frame-id')
        if not isinstance(page_no, int):
            errors.append(f'{path.name}: pid.page-no must be int')
        if not isinstance(frame_id, str):
            errors.append(f'{path.name}: pid.frame-id must be string')
        elif not re.fullmatch(r'[a-z]', frame_id):
            errors.append(f'{path.name}: pid.frame-id must match [a-z]')

        if isinstance(page_no, int):
            pages_present.add(page_no)

        title = obj.get('title')
        content = obj.get('content')

        if stem in HEADER_KEYS:
            if not isinstance(title, dict):
                errors.append(f'{path.name}: header frame must include title object')
            else:
                if title.get('type') != 'markup':
                    errors.append(f'{path.name}: title.type must be markup')
                tdata = title.get('data')
                if not isinstance(tdata, str):
                    errors.append(f'{path.name}: title.data must be string')
                elif '\r\n' not in tdata:
                    errors.append(f'{path.name}: title.data must contain CRLF')
                else:
                    for i, ln in enumerate(lines(tdata), 1):
                        v = visible_len(ln)
                        if v > LINE_WIDTH:
                            errors.append(f'{path.name}: title line {i} width {v} exceeds {LINE_WIDTH}')
        else:
            if title is not None:
                errors.append(f'{path.name}: non-header frame should omit title')

        if not isinstance(content, dict):
            errors.append(f'{path.name}: missing content object')
            continue

        if content.get('type') != 'markup':
            errors.append(f'{path.name}: content.type must be markup')

        cdata = content.get('data')
        if not isinstance(cdata, str):
            errors.append(f'{path.name}: content.data must be string')
            continue

        if '\r\n' not in cdata:
            errors.append(f'{path.name}: content.data must contain CRLF')
        if NAV_TEXT not in cdata:
            errors.append(f'{path.name}: missing nav line text')

        for i, ln in enumerate(lines(cdata), 1):
            v = visible_len(ln)
            if v > LINE_WIDTH:
                errors.append(f'{path.name}: content line {i} width {v} exceeds {LINE_WIDTH}')

        total_rows = row_count(cdata)
        if isinstance(title, dict):
            total_rows += row_count(title.get('data', ''))
        if total_rows > MAX_ROWS:
            errors.append(f'{path.name}: total rows {total_rows} exceeds {MAX_ROWS}')

        routing = obj.get('routing-table')
        if routing is not None:
            if stem not in ROUTED_ONLY:
                errors.append(f'{path.name}: routing-table only allowed on intro/menu/index/help frames')
            if not isinstance(routing, list) or len(routing) != 11:
                errors.append(f'{path.name}: routing-table must be 11-item list')
            elif not all(isinstance(v, int) for v in routing):
                errors.append(f'{path.name}: routing-table entries must be integers')

    for req in REQUIRED:
        if req not in frame_keys:
            errors.append(f'Missing required frame {req}.json')

    for path, obj in parsed.items():
        routing = obj.get('routing-table')
        if not isinstance(routing, list) or len(routing) != 11:
            continue
        for idx, val in enumerate(routing):
            if val == 0:
                continue
            if val not in pages_present:
                errors.append(f'{path.name}: route index {idx} references missing page {val}')

    if errors:
        for err in errors:
            print(err)
        sys.exit(1)

    print(f'OK: validated {len(files)} frame files.')


if __name__ == '__main__':
    main()
