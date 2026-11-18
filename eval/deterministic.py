import re


# Multiple-choice: option letters in the question and in the response
_MCQ_SPLIT = re.compile(r'(?:^|\n|\s)\(?([A-J])[.)]\s+')
_ANSWER_FALLBACK = re.compile(r'\*\*\(?([A-J])[.\):]')

_L = r'([A-J])'
_CUE_GAP = r'\s*(?:\*\*|>|\s|[:\-\u2014\u2013])*\s*'
_MCQ_EXPLICIT = [
    re.compile(r'\\boxed\{\s*\(?' + _L + r'\)?\s*\}', re.I),
    re.compile(r'(?:final\s+answer|correct\s+answer|correct\s+option|correct\s+choice|best\s+choice|best\s+option|right\s+choice|answer\s+is|choice\s+is|answer)' + _CUE_GAP + r'\(?' + _L + r'(?:[.\)\*:\s]|$)', re.I),
    re.compile(r'answer\b[^.\n]{0,80}?\bis\s*(?:\*\*)?\s*\(?' + _L + r'(?:[.\)\*:\s,]|$)', re.I),
    re.compile(r'option\s*(?:\*\*)?\s*\(?' + _L + r'\)?(?:\*\*)?\s+is\s+(?:the\s+)?(?:correct|right|exception|answer|best|most\s+likely)', re.I),
    re.compile(r'(?:^|[\s(*])' + _L + r'(?:\*\*|\))?\s+is\s+the\s+(?:correct|right|best|most\s+(?:likely|appropriate)|only\s+(?:false|true|correct))\b', re.I | re.M),
    re.compile(r'(?:^|\n)\s*(?:\*\*)?\(?' + _L + r'[.)]?(?:\*\*)?\s*$', re.I),
]
_MCQ_MULTI = re.compile(r'(?:final\s+answer|answer)\s*[:\-]?\s*(?:\*\*|>|\s)*\s*\(?[A-J]\)?\s*(?:,|and|&|/)\s*\(?[A-J]\b', re.I)
_LETTER_LINE = re.compile(r'^\s*[-\u2022*]*\s*\(?([a-j])[.):]\s*(.*?)\s*$', re.I)

# Multiple-choice: answers given by the option text instead of the letter
_TXT_ART = r'(?:(?:the|a|an)\W+)?'
_TXT_NEG = re.compile(r"(?:\bnot|n't|\bnever|\bexcept|\brather than|\binstead of|\bnor|\bunlikely|\bless likely|\bno longer)\W+(?:(?:the|a|an|option|to be|be|for|of)\W+){0,2}$")
_TXT_HYPO = re.compile(r"(?:\bif\b|\bunless\b|\bassum|\bsuppos|\bmaybe\b|\bperhaps\b|\bmight\b|\bcould\b|\bpossibl|\bin that case\b|\bthen\b|\bimply|\bimplies|\bexpected\b|\bi recall\b|\bi've seen\b|\bi have seen\b|\bsome (?:sources|exams)|\bwould be if\b|\bwhether\b)[^.\n]{0,60}$")
_TXT_CUE = re.compile(r"(?:(?:the\s+)?(?:final|correct|best|right|most\s+likely|most\s+appropriate)\s+answer|(?:the\s+)?answer|(?:the\s+)?(?:correct|best|right)\s+(?:choice|option)|(?:the\s+)?(?:most\s+likely\s+|likely\s+|probable\s+|correct\s+|final\s+)?diagnosis|(?:the\s+)?correct\s+response)\s*(?:is|would\s+be|should\s+be|:|-|\u2014|\u2013)\s*")
_TXT_FILL = r"(?:(?:the|a|an|option|choice|that|likely|most\s+likely|probably|clearly|definitely|therefore|thus|then|of)\W+){0,3}"
_TXT_ISCUE = r"(?:\bis|\bare|\bwould\s+be|\bshould\s+be|\bwas)\s+(?:(?:most\s+likely|likely|probably|clearly|consistent\s+with|suggestive\s+of|indicative\s+of|diagnostic\s+of|characteristic\s+of|typical\s+of|due\s+to|caused\s+by)\s+)?"
_TXT_LASTLINE_PREFIX = re.compile(r"^(?:(?:therefore|thus|so|hence|overall|in\s+conclusion)\W+)?(?:the\s+)?(?:final\s+answer|answer|correct\s+answer|correct\s+choice|best\s+answer|most\s+likely\s+diagnosis|diagnosis)\s*(?:(?:is|would\s+be|:|-|\u2014|\u2013)\s*){0,2}(?:the\s+)?")
_TXT_SYN = {"true": "(?:true|yes)", "false": "(?:false|no)"}

# Yes/no answers
_YES = re.compile(r'\byes\b', re.I)
_NO = re.compile(r'\bno\b', re.I)

_YES_ALT = re.compile(r'\b(?:true|correct|present)\b', re.I)
_NO_ALT = re.compile(r'\b(?:false|incorrect|absent)\b', re.I)
_ALT_ANY = re.compile(r'\b(?:true|correct|present|false|incorrect|absent)\b', re.I)
_NEGATOR = re.compile(r'\b(?:not|never|nor|without|cannot|isn|doesn|didn|wasn|don)\b', re.I)
_BOOL_GT = {"yes": True, "true": True, "no": False, "false": False}

# Short answers
_NUMWORD = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def _norm(s):
    return re.sub(r'[^a-z0-9 ]+', ' ', str(s).lower()).strip()


def _norm_num(s):
    """Normalize text and convert spelled-out numbers to digits."""
    return ' '.join(_NUMWORD.get(w, w) for w in _norm(s).split())


_NONE_WORDS = re.compile(r'\b(?:no|none|zero|absent|nil)\b')


def _matches_short(gt_n, resp_n):
    """Whole-word match of a one-word ground truth in a short response, tolerant to plurals and "no X" for zero."""
    if gt_n == "0" and not re.search(r'\d', resp_n) and _NONE_WORDS.search(resp_n):
        return True
    forms = {gt_n, gt_n + "s", gt_n + "es"}
    if gt_n.endswith("s") and len(gt_n) > 3:
        forms.add(gt_n[:-1])
    return any(re.search(r'\b%s\b' % re.escape(f), resp_n) for f in forms)


def parse_choices(question):
    """Parse the options of a question into {letter: normalized option text}."""
    m = re.search(r'Choices?\s*:\s*(.*)$', str(question), re.S | re.I)
    if not m:
        return {}
    parts = _MCQ_SPLIT.split(m.group(1))
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i].upper()] = _norm(parts[i + 1])
    return out


_PUBMED_GT = {"yes": "true", "no": "false", "maybe": "maybe"}


def ground_truth_letter(choices, answer, answer_idx=None):
    """Return the option letter of the ground truth, or None if it is ambiguous."""
    a = _norm(answer)
    first = str(answer).strip().split("\n", 1)[0].strip().lower()
    if first in _PUBMED_GT and any(t in ("true", "false") for t in choices.values()):
        a = _PUBMED_GT[first]
    by_text = None
    if a:
        exact = [letter for letter, text in choices.items() if text == a]
        if len(exact) == 1:
            by_text = exact[0]
        elif not exact:
            hits = [letter for letter, text in choices.items() if a in text or text in a]
            if len(hits) == 1:
                by_text = hits[0]
    if by_text is not None and list(choices.values()).count(choices[by_text]) > 1:
        return None
    idx = str(answer_idx).strip().upper() if answer_idx not in (None, "") else ""
    if idx in choices:
        if by_text is None:
            return idx
        if idx != by_text:
            return None
    return by_text


def _letter_line_form(text, choices):
    """Return the letter of a closing line like "C. <option text>" if it is the only exact restatement."""
    exact, led = [], []
    for line in re.sub(r'[*_`>]+', '', text).split("\n"):
        m = _LETTER_LINE.match(line)
        if not m or m.group(1).upper() not in choices:
            continue
        letter = m.group(1).upper()
        led.append(letter)
        rx = _option_rx(choices[letter])
        if rx and re.fullmatch(r'(?:(?:the|a|an)\W+)?' + rx + r'\W*', m.group(2).lower()):
            exact.append(letter)
    if len(exact) == 1 and led[-1] == exact[0]:
        return exact[0]
    return None


def predicted_letter(response, choices):
    """Return the option letter stated in the response, "MULTI" if several are stated, or None."""
    text = str(response).strip()
    if _MCQ_MULTI.search(text[-300:]):
        return "MULTI"
    best = None
    for rx in _MCQ_EXPLICIT:
        for m in rx.finditer(text):
            letter = m.group(1).upper()
            if letter in choices and (best is None or m.start() >= best[0]):
                best = (m.start(), letter)
    if best:
        return best[1]
    bold = {m.upper() for m in _ANSWER_FALLBACK.findall(text) if m.upper() in choices}
    if len(bold) == 1:
        return next(iter(bold))
    return _letter_line_form(text, choices)


def _option_rx(opt):
    """Build a regex that matches the text of one option inside a cleaned response."""
    ws = opt.split()
    if len(ws) >= 2 and ws[0] in ("the", "a", "an"):
        ws = ws[1:]
    if not ws or (len(ws) == 1 and ws[0] in ("the", "a", "an", "of", "and", "or", "is", "all", "none", "both")):
        return None
    parts = []
    for w in ws:
        if len(ws) == 1 and w in _TXT_SYN:
            parts.append(_TXT_SYN[w])
        else:
            parts.append(re.escape(w) + ("s?" if len(w) > 3 and not w.endswith("s") else ""))
    return "(?<![a-z0-9])" + (r"\W+" + _TXT_ART).join(parts) + "(?![a-z0-9])"


def _clean_text(t):
    t = re.sub(r"<answer>|</answer>", "\n", str(t))
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"\([^()]{0,80}\)", " ", t)
    return re.sub(r"[ \t]+", " ", t.lower())


def _option_spans(text, rxs):
    """Return the letters of all options found in the text, ignoring matches nested in a longer option."""
    out = [(m.start(), m.end(), l) for l, r in rxs.items() for m in re.finditer(r, text)]
    return {s[2] for s in out if not any(o != s and o[0] <= s[0] and s[1] <= o[1] and (o[1] - o[0]) > (s[1] - s[0]) for o in out)}


def predicted_option_text(response, choices):
    """Return the option letter stated by its text when no letter is given, or None."""
    txt = _clean_text(response).strip()
    rxs = {l: _option_rx(o) for l, o in choices.items()}
    rxs = {l: r for l, r in rxs.items() if r}
    if not rxs:
        return None
    evidence = set()
    if len(re.findall(r"[a-z0-9]+", txt)) <= 12 and not re.search(r"\bnot\b|n't|\bnever\b|\bexcept\b|\bor\b", txt):
        hit = _option_spans(txt, rxs)
        if len(hit) == 1:
            evidence |= hit
    for l, r in rxs.items():
        for m in re.finditer(r + r"\s+(?:is|was|would\s+be)\s+(?:the\s+)?(?:correct|right|best|most\s+likely|most\s+appropriate|most\s+probable)(?:\s+(?:answer|option|choice|diagnosis))?", txt):
            if _TXT_NEG.search(txt[max(0, m.start() - 30):m.start()]) or _TXT_HYPO.search(txt[max(0, m.start() - 60):m.start()]):
                continue
            s0 = max(txt.rfind(".", 0, m.start()), txt.rfind("\n", 0, m.start())) + 1
            s1 = re.search(r"[.!?\n]", txt[m.end():])
            sent = txt[s0:m.end() + (s1.start() if s1 else 0)]
            if _option_spans(sent, rxs) == {l}:
                evidence.add(l)
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    if lines:
        ll = _TXT_LASTLINE_PREFIX.sub("", lines[-1])
        ll = re.sub(r"^[-\u2022]\s*", "", ll).strip(" .!\"'\u201c\u201d\u2018\u2019")
        for l, r in rxs.items():
            if re.fullmatch(r"(?:(?:the|a|an)\W+)?" + r + r"\W*", ll):
                evidence.add(l)
    first = re.split(r"(?<=[.!?])\s+|\n", txt, maxsplit=1)[0][:400]
    if not re.search(r"\?|\bnot\b|n't|\bexcept\b|\bor\b|\bvs\b|\bversus\b|\b(?:if|unless|assum\w*|suppos\w*|maybe|perhaps|might|could|possibl\w*|whether)\b", first):
        hit = set()
        for l, r in rxs.items():
            if re.search(_TXT_ISCUE + r"(?:(?:the|a|an)\W+)?" + r, first):
                hit.add(l)
            if re.match(r"^\W*(?:(?:the|a|an)\W+)?" + r + r"\W+(?:is|are|was)\W+(?:the|a|an|also|characterized|defined|known)\b", first):
                hit.add(l)
        if len(hit) == 1 and len(_option_spans(first, rxs)) == 1:
            evidence |= hit
    if len(evidence) != 1:
        return None
    veto = None
    for m in _TXT_CUE.finditer(txt):
        before = txt[max(0, m.start() - 60):m.start()]
        if _TXT_HYPO.search(before) or _TXT_NEG.search(before[-30:]) or "differential" in before[-15:]:
            continue
        seg = txt[m.end():m.end() + 300]
        stop = re.search(r"[.!?\n]", seg)
        seg = seg[:stop.start() + 1] if stop else seg
        if "?" in seg:
            continue
        after = re.sub(r'^["\'\u201c\u2018]+\s*', "", seg)
        if not re.match(_TXT_FILL + r"(?:" + "|".join(rxs.values()) + ")", after) or re.match(_TXT_FILL + r"(?:not|neither|none)\b", after):
            continue
        hit = _option_spans(after, rxs)
        veto = next(iter(hit)) if len(hit) == 1 else "MANY"
    if veto is not None and veto != next(iter(evidence)):
        return None
    return next(iter(evidence))


_FINAL_MARK = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|my\s+answer\s+is|correct\s+answer|answer\s*[:\-]|"
    r"conclusion\s*:|verdict\s*:|in\s+summary|\\boxed\{|<answer>)", re.I)


def binary_score(response, gt, exact_max_words=25, scope_words=30, final_words=12):
    """Score a yes/no answer from the final answer statement, then the opening words, then the whole text."""
    text = str(response)
    fin = final_words
    if fin and fin > 0:
        last = None
        for m in _FINAL_MARK.finditer(text):
            last = m
        if last is not None:
            tail = " ".join(text[last.end():].split()[:fin])
            tail_yes, tail_no = _YES.search(tail) is not None, _NO.search(tail) is not None
            if tail_yes and tail_no:
                return 0.0
            if tail_yes != tail_no:
                return 1.0 if tail_yes == _BOOL_GT.get(_norm(gt), True) else 0.0
    scope = scope_words
    if scope and scope > 0:
        head = " ".join(text.split()[:scope])
        head_yes, head_no = _YES.search(head) is not None, _NO.search(head) is not None
        if head_yes and head_no:
            return 0.0
        if head_yes != head_no:
            return 1.0 if head_yes == _BOOL_GT.get(_norm(gt), True) else 0.0
    says_yes = _YES.search(text) is not None
    says_no = _NO.search(text) is not None
    if not says_yes and not says_no:
        short = _norm(text)
        if len(short.split()) <= exact_max_words:
            hit = _ALT_ANY.search(short)
            if hit and not _NEGATOR.search(short[:hit.start()]):
                says_yes = _YES_ALT.search(short) is not None
                says_no = _NO_ALT.search(short) is not None
    if says_yes == says_no:
        return 0.0
    return 1.0 if says_yes == _BOOL_GT.get(_norm(gt), True) else 0.0


def deterministic_score(sample, exact_max_words=25, binary_scope_words=30, binary_final_words=12):
    """Score a sample with fixed rules, returns (score, method) or (None, None) if the judge is needed."""
    if 'mimic' in str(sample.get('id', '')):
        return None, None

    question = sample.get('question', '')
    gt = sample.get('answer', '')
    response = sample.get('response', '')
    if not str(response).strip():
        return None, None

    choices = parse_choices(question)
    if len(choices) >= 2:
        gl = ground_truth_letter(choices, gt, sample.get('answer_idx'))
        if gl is not None:
            pl = predicted_letter(response, choices)
            if pl is not None:
                return (1.0 if pl == gl else 0.0), 'mcq_letter'
            pl = predicted_option_text(response, choices)
            if pl is not None:
                return (1.0 if pl == gl else 0.0), 'mcq_text'
        return None, None

    if _norm(gt) in _BOOL_GT:
        return binary_score(response, gt, exact_max_words, binary_scope_words, binary_final_words), 'binary'

    gt_n = _norm_num(gt)
    if gt_n and len(gt_n.split()) == 1:
        resp_n = _norm_num(response)
        if resp_n and len(resp_n.split()) <= exact_max_words:
            return (1.0 if _matches_short(gt_n, resp_n) else 0.0), 'exact_short'

    return None, None
