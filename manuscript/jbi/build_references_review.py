"""Build references_review.csv: one row per bib entry for human review.

Usage: python3 build_references_review.py .   (from manuscript/jbi, after compile.sh)
Verdicts and open issues are hard-coded below from CITATION_REVIEW.md; update both together.
"""
import csv, re, sys, unicodedata
from pathlib import Path

J = Path(sys.argv[1])  # manuscript/jbi
bib = (J / 'references.bib').read_text()
review = (J / 'CITATION_REVIEW.md').read_text()


def fields(body):
    out = {}
    for m in re.finditer(r'^\s*(\w+)\s*=\s*\{', body, re.M):
        i, depth, j = m.end(), 1, m.end()
        while depth:
            depth += {'{': 1, '}': -1}.get(body[j], 0); j += 1
        out[m.group(1).lower()] = re.sub(r'\s+', ' ', body[i:j - 1]).strip()
    return out


def detex(s):
    s = re.sub(r'\\(?:emph|textit|textbf|texttt)\{([^}]*)\}', r'\1', s or '')
    acc = {"'": '\u0301', '`': '\u0300', '"': '\u0308', '^': '\u0302', '~': '\u0303', '=': '\u0304', '.': '\u0307'}
    s = re.sub(r'\\i\b', 'ı', s)
    s = re.sub(r"\{?\\(['`\"^~=.])\{?(\w)\}?\}?", lambda m: unicodedata.normalize('NFC', m.group(2) + acc[m.group(1)]), s)
    s = s.replace('\\&', '&').replace('--', '–').replace('~', ' ').replace('\\%', '%')
    s = re.sub(r'\\(?:i)\b', 'i', s)
    return re.sub(r'\s+', ' ', re.sub(r'[{}]', '', s)).strip()


entries = {}
for t, k, b in re.findall(r'^@(\w+)\{([^,]+),(.*?)\n\}', bib, re.M | re.S):
    entries[k] = dict(fields(b), _type=t)

# ── where each key is cited: inline \input tables, track section numbers ──────
main = (J / 'main.tex').read_text()
aux = (J / 'main.aux').read_text()
tabnum = dict(re.findall(r'\\newlabel\{(tab:[^}]+)\}\{\{([^}]+)\}', aux))


def expand(text):
    def rep(m):
        p = J / (m.group(1) if m.group(1).endswith('.tex') else m.group(1) + '.tex')
        if not p.exists():
            return ''
        body = p.read_text()
        lab = re.search(r'\\label\{(tab:[^}]+)\}', body)
        tag = f'Table {tabnum.get(lab.group(1), "?")}' if lab else p.stem
        return f'\x00TAB {tag}\x00' + body + '\x00ENDTAB\x00'
    return re.sub(r'\\input\{([^}]+)\}', rep, text)


text = re.sub(r'(?m)(?<!\\)%.*$', '', expand(main))
text = text.split('\\begin{document}', 1)[1]
marker = re.compile(r'\x00TAB ([^\x00]+)\x00|\x00ENDTAB\x00|\\(sub)?section(\*?)\{([^}]*)\}|\\appendix|\\begin\{abstract\}')
locs, sec, sub, appendix, cur, table = [], 0, 0, False, '', None
for m in marker.finditer(text):
    tok = m.group(0)
    if m.group(1):
        table = m.group(1)[4:] if m.group(1).startswith('TAB ') else m.group(1)
    elif tok == '\x00ENDTAB\x00':
        table = None
    elif tok == '\\appendix':
        appendix, sec = True, 0
    elif tok == '\\begin{abstract}':
        cur = 'Abstract'
    else:
        title = detex(m.group(4))
        if m.group(3):
            cur = title
        elif m.group(2):
            sub += 1; cur = f'{sec if not appendix else chr(64 + sec)}.{sub} {title}'
        else:
            sec += 1; sub = 0; cur = f'{sec if not appendix else "Appendix " + chr(64 + sec)} {title}'
    locs.append((m.end(), table or cur))


def location(pos):
    out = ''
    for end, loc in locs:
        if end > pos:
            break
        out = loc
    return out


BOUND_L = re.compile(r'(?:(?<=[a-z0-9)\]}])\.\s+(?=[A-Z\\(]))|&|\\\\|\n\s*\n|\\item|\\caption\{|\\paragraph\{[^}]*\}|\x00[^\x00]*\x00')
BOUND_R = re.compile(r'(?:(?<=[a-z0-9)\]}])\.(?=\s+[A-Z\\(]|\s*$))|&|\\\\|\n\s*\n|\\item|\x00')


SECTION = r'\\(?:sub)*section\*?\{[^}]*\}(?:\s*\\label\{[^}]*\})?'
BOUND_L = re.compile(r'(?:(?<=[a-z0-9)\]}])\.\s+(?=[A-Z\\(]))|&|\\\\|\n\s*\n|\\item|\\caption\{|\\paragraph\{[^}]*\}|\x00[^\x00]*\x00|' + SECTION)
BOUND_R = re.compile(r'(?:(?<=[a-z0-9)\]}])\.(?=\s+[A-Z\\(]|\s*$))|&|\\\\|\n\s*\n|\\item|\x00')
ROW_L = re.compile(r'\\\\|\\hline|\x00[^\x00]*\x00')
ROW_R = re.compile(r'\\\\|\x00')


def clean(frag):
    frag = re.sub(r'\\\\ ', ' ', frag)
    frag = re.sub(r'\\(?=\s)', ' ', frag.replace('\\times', '×').replace('\\,', ' ').replace('{,}', ','))
    frag = re.sub(r'\\cite[pt]?\{[^}]+\}', '[cite]', frag)
    frag = re.sub(r'\s*\((?:Figure|Table|Section|Appendix)s?~?\\ref\{[^}]*\}\)', '', frag)
    frag = re.sub(r'(?:Figure|Table|Section|Appendix|\\S)~?\\ref\{[^}]*\}', '[ref]', frag)
    frag = re.sub(r'\\(?:begin|end)\{[^}]*\}|\\(?:hline|label\{[^}]*\}|cline\{[^}]*\}|multicolumn\{[^}]*\}\{[^}]*\})', ' ', frag)
    frag = re.sub(r'\$([^$]*)\$', r'\1', frag)
    frag = frag.replace('&', ' | ')
    frag = re.sub(r'\\[a-zA-Z]+\*?(?=\{)', '', frag)
    frag = re.sub(r'\\[a-zA-Z]+\*?|\\\\', ' ', frag)
    frag = detex(frag).strip(' |')
    return re.sub(r'\s+([,.;)])', r'\1', frag)


def sentence(pos, end):
    in_table = location(pos).startswith('Table')
    lb, rb = (ROW_L, ROW_R) if in_table else (BOUND_L, BOUND_R)
    left = [m.end() for m in lb.finditer(text, max(0, pos - 1500), pos)]
    a = left[-1] if left else max(0, pos - 400)
    r = rb.search(text, end, end + 1500)
    b = (r.start() if in_table else r.end()) if r else end + 300
    frag = clean(text[a:b])
    return frag[:450] + ('…' if len(frag) > 450 else '')


where, claims = {}, {}
for m in re.finditer(r'\\cite[pt]?\{([^}]+)\}', text):
    loc = location(m.start())
    for k in (x.strip() for x in m.group(1).split(',')):
        where.setdefault(k, [])
        if loc not in where[k]:
            where[k].append(loc)
        claims.setdefault(k, sentence(m.start(), m.end()))

# ── bbl order = printed reference number ───────────────────────────────────────
bbl_order = re.findall(r'\\bibitem\{([^}]+)\}', (J / 'main.bbl').read_text())

# ── verdicts: August review (Parts 1–2), then the 2026-10-26 pass ─────────────
verdict, fulltext = {}, {}
for part in review.split('## Part')[1:3]:
    for line in part.splitlines():
        m = re.search(r'`([a-z]+\d{4}[a-z0-9]*)`', line)
        if line.startswith('|') and m:
            c = [x.strip() for x in line.split('|')[1:-1]]
            verdict[m.group(1)] = ('2026-08-22', re.sub(r'\*', '', c[-1]))
            u = re.search(r'\]\((https?://[^)]+)\)', c[1])
            if u:
                fulltext[m.group(1)] = u.group(1)
SURVEY = set(re.findall(r'`([a-z]+\d{4}[a-z0-9]*)`', review.split('## Part 2')[1].split('## Part 3')[0]))
P3 = '2026-10-26'
for k in ['adamic2003friends', 'barabasi1999scaling', 'liben2007link', 'zhou2009local', 'katz1953status',
          'chen2020mirdb', 'czi2025cellxgene', 'li2014hmdd', 'lu2015spm', 'chen2018bnpmda', 'qu2018llcmda',
          'yu2020tcrwmda', 'wang2014hnm', 'tong2006rwr']:
    verdict[k] = (P3, '✅ Supported — the original work for the name or resource it is cited for.')
verdict['hu2020ogb'] = (P3, '✅ Supported (after fix) — now cited only for the datasets and their official negatives. '
                        'Was ❌ for "OGB already reports such heuristics": the paper reports none; moved to ogb2026leaderboard.')
verdict['ogb2026leaderboard'] = (P3, '✅ Supported — ranks re-read 2026-10-26 (ddi 30th of 32; ppa 14th of 27, BUDDY 13th); '
                                 'heuristic rows submitted by third parties in 2021.')

NOTES = {
    'briere2026kgeleakage': 'OPEN: the sentence says the study\'s permutation test found no evidence of degree exploitation. '
                            'Not re-confirmed from full text (bioRxiv 403 in Aug, 429 on 2026-10-26). Needs someone with access to read v3. '
                            'Still a preprint (v3, 2026-07-22).',
    'toutanova2015observed': 'Claim (inverse-relation leak in FB15k) is in the paper body, not the abstract; no verbatim quote taken.',
    'kotnis2017analysis': 'Shows sampling-method × dataset bias, not degree bias as such; cited for "bias in link prediction generally". '
                          'KBCOM\'18 venue confirmed (Heidelberg CL news, best paper).',
    'barabasi2004networkbiology': 'No abstract exists; support is title/scope level.',
    'krichene2020sampled': 'About evaluation-time sampled metrics in recommenders; the sentence hedges with "likely rooted in".',
    'huang2023gelato': 'arXiv-only, no peer-reviewed venue found (2026-10-26).',
    'aiyappa2024implicit': 'arXiv-only (2026-10-26).',
    'zeng2025gonnmda': 'Crossref title reads "A Ordered"; bib keeps "An". Decide at proofs.',
    'zhu2023pitfalls': 'Aug review checked the term "target-link inclusion"; the text now cites it only for "actively re-examined".',
    'almusawi2023review': 'In the bib but not cited; not printed.',
    'yun2021neognn': 'In the bib but not cited; not printed.',
}
LINK_FALLBACK = {'hu2020ogb': 'https://arxiv.org/abs/2005.00687', 'li2023evaluating': 'https://arxiv.org/abs/2306.10453',
                 'zhang2018seal': 'https://arxiv.org/abs/1802.09691', 'zhu2021nbfnet': 'https://arxiv.org/abs/2106.06935',
                 'chamberlain2023buddy': 'https://arxiv.org/abs/2209.15486', 'yun2021neognn': 'https://arxiv.org/abs/2206.04216'}


def metadata(k, e):
    if k == 'ogb2026leaderboard':
        return 'Web page, read 2026-10-26'
    if k in ('tian2024mgcnss', 'jin2022nimgsa', 'yu2022hgdti', 'cao2024ngcn', 'zhang2024kmerpmtf',
             'toutanova2015observed', 'krichene2020sampled'):
        return 'Crossref, DOI added 2026-10-26'
    if k in ('zhang2018seal',):
        return 'Semantic Scholar (NeurIPS 2018, pp. 5171–5181)'
    if k in ('zhu2021nbfnet', 'chamberlain2023buddy'):
        return 'arXiv comment / journal_ref'
    if k == 'kotnis2017analysis':
        return 'Authors\' institution (KBCOM\'18); arXiv'
    if k in ('huang2023gelato', 'aiyappa2024implicit', 'almusawi2023review'):
        return 'arXiv API (no published version)'
    if k == 'briere2026kgeleakage':
        return 'bioRxiv API (v3, no published version)'
    if k in ('hu2020ogb', 'li2023evaluating', 'yun2021neognn'):
        return 'DBLP (2026-08-13 note); not re-checked 2026-10-26'
    return 'Crossref (DOI), matches'


def short_ref(e):
    authors = [a.strip() for a in detex(e.get('author', '')).split(' and ')]
    first = authors[0].split(',')[0] if authors and authors[0] else ''
    who = first + (' et al.' if len(authors) > 2 else (' & ' + authors[1].split(',')[0] if len(authors) == 2 else ''))
    venue = detex(e.get('journal') or e.get('booktitle') or e.get('howpublished') or '') or e.get('url', '')
    loc = ' '.join(x for x in [e.get('volume', ''), f"({e['number']})" if e.get('number') else '',
                               detex(e.get('pages', ''))] if x)
    return f"{who} ({e.get('year', '')}). {detex(e.get('title'))}. {venue}{', ' + loc if loc else ''}."


rows = []
keys = bbl_order + [k for k in entries if k not in bbl_order]
for k in keys:
    e = entries[k]
    doi = e.get('doi', '')
    link = (f'https://doi.org/{doi}' if doi else e.get('url') or
            (f"https://arxiv.org/abs/{e['eprint']}" if e.get('eprint') else '') or LINK_FALLBACK.get(k, ''))
    if not link:
        m = re.search(r'arXiv:(\d{4}\.\d{4,5})', e.get('journal', ''))
        link = f'https://arxiv.org/abs/{m.group(1)}' if m else ''
    v_date, v = verdict.get(k, ('', ''))
    note = NOTES.get(k, '')
    if k in SURVEY:
        note = (note + ' ' if note else '') + ('Surveyed paper: its Table 3 classification lives in '
                                               'results/literature_survey_adjudication.tsv; the Aug review\'s survey columns are superseded.')
    rows.append({
        'ref_no': bbl_order.index(k) + 1 if k in bbl_order else '',
        'bib_key': k,
        'reference': short_ref(e),
        'link': link,
        'fulltext_link': fulltext.get(k, '') if fulltext.get(k, '') != link else '',
        'cited_in': '; '.join(where.get(k, [])) or 'not cited',
        'claim_in_paper': claims.get(k, ''),
        'metadata_check': metadata(k, e),
        'content_verdict': v,
        'verdict_date': v_date,
        'open_issue': note,
        'reviewed_by': '', 'review_date': '', 'reviewer_ok': '', 'reviewer_comment': '',
    })

with open(J / 'references_review.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), 'rows')
