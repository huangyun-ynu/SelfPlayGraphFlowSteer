"""Public-text provenance and conservative, concrete delegation boundaries.

This module never receives references or evaluator results. It compares surface
representations only; it does not solve a problem or infer new mathematical facts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

MAX_ISSUES = 8
_SYMBOL = r"(?:[A-Z]{1,4}|[a-z])(?:_\{?[A-Za-z0-9]+\}?)?"
_DIAGRAM = re.compile(r"\[asy\].*?\[/asy\]", re.I | re.S)
_EQUALITY = re.compile(r"(?<![<>=!])=(?!=)|\b(?:equals|is equal to)\b", re.I)
_MATH_TOKEN = re.compile(r"(?:root|log)_\d+|\\[A-Za-z]+|\\[{}]|[A-Za-z]+\d*|\d+(?:\.\d+)?|[^\s]")
_PROSE_TOKENS = frozenset({'a', 'an', 'and', 'the', 'let', 'for', 'to', 'of', 'in',
                           'is', 'are', 'not', 'all', 'can', 'use', 'end', 'be', 'or',
                           'set', 'has', 'if', 'its', 'one', 'two', 'six', 'our', 'any'})


@dataclass(frozen=True)
class Fragment:
    text: str
    start_offset: int
    end_offset: int

    def group(self) -> str:
        return self.text[self.start_offset:self.end_offset]

    def span(self) -> tuple[int, int]:
        return self.start_offset, self.end_offset


def _math_token(token: str, *, nested: bool) -> bool:
    if token in {'+', '-', '*', '/', '^', '_', '!', '(', ')', '[', ']', '{', '}', '|', ':', "'"}:
        return True
    if token == ',':
        return nested
    if token.startswith('\\'):
        return len(token) > 1 and token not in {'\\text', '\\quad', '\\qquad', '\\begin', '\\end'}
    if re.fullmatch(r'\d+(?:\.\d+)?', token):
        return True
    if token.lower() in {'sqrt', 'sin', 'cos', 'tan', 'atan', 'asin', 'acos', 'ceil', 'floor', 'log', 'ln', 'exp', 'pi', 'dp', 'angle', 'mod', 'perimeter', 'area', 'volume', 'length'} or re.fullmatch(r'(?:root|log)_\d+|[A-Za-z]\d+', token):
        return True
    return (len(token) == 1 or (len(token) <= 3 and token.lower() not in _PROSE_TOKENS)) and token.isalpha()


def equalities(text: str) -> list[Fragment]:
    """Extract complete surface equations, with balanced groups and no evaluation.

    In particular, x+y=1 is not reduced to y=1, and vectors/sets keep commas.
    Unknown prose is a boundary, not a guessed mathematical operator.
    """
    tokens = list(_MATH_TOKEN.finditer(text))
    result = []
    for eq in _EQUALITY.finditer(text):
        before = [t for t in tokens if t.end() <= eq.start()]
        after = [t for t in tokens if t.start() >= eq.end()]
        bounds = []
        for side, direction in [(before[::-1], -1), (after, 1)]:
            stack = []
            boundary = None
            for t in side:
                token = t.group().replace('\\{', '{').replace('\\}', '}')
                openers, closers = ('([{', ')]}') if direction == 1 else (')]}', '([{')
                if token in closers:
                    if not stack:
                        break
                    if openers.index(stack[-1]) != closers.index(token):
                        break
                    stack.pop()
                elif token in openers:
                    stack.append(token)
                if not _math_token(token, nested=bool(stack)):
                    break
                boundary = t.start() if direction == -1 else t.end()
            # A partial/unclosed expression is ambiguous, not a hard assertion.
            bounds.append(boundary if not stack else None)
        if all(bound is not None for bound in bounds):
            while text[bounds[0]] in ': \t':
                bounds[0] += 1
            fragment = Fragment(text, bounds[0], bounds[1])
            lhs, rhs = text[bounds[0]:eq.start()].strip(), text[eq.end():bounds[1]].strip()
            if lhs and rhs and any(c.isalpha() for c in lhs) and not rhs.endswith(('+', '-', '*', '/', '^')):
                result.append(fragment)
    return result
_ANSWER = re.compile(
    r"\b(?:final\s+answer|candidate\s+answer|answer)\s*(?:is|=|:|equals)\s*"
    r"[^;\n,]+|答案\s*(?:是|=|:|：)\s*[^;\n,]+", re.I,
)
_VALUE_STATEMENT = re.compile(
    rf"(?<![\w])(?-i:{_SYMBOL})\s+is\s+[+-]?\d+(?:\.\d+)?\b|"
    r"\b(?:path count|number of paths|computed value|square root)\s+is\s+[+-]?\d+(?:\.\d+)?\b|"
    r"\b(?:return|report|output|submit)\s+(?:exactly\s+)?[+-]?\d+(?:\.\d+)?\b", re.I,
)
_ALGORITHM = re.compile(
    r"\b(?:dynamic programming|(?:depth|breadth)[ -]first search|"
    r"binary search|inclusion[ -]exclusion|generating functions?|"
    r"transfer matrix|matrix exponentiation|Gaussian elimination|"
    r"Euclidean algorithm|Newton(?:'s)? method|Monte Carlo|"
    r"memoized? recursion|backtracking|brute[ -]force enumeration|"
    r"quadratic formula|Chinese remainder theorem|Pythagorean theorem|"
    r"similar triangles|law of (?:cosines|sines)|prime factorization)\b", re.I,
)
_METHOD_DIRECTIVE = re.compile(
    r"\b(?:use|apply|implement|perform|execute|solve\s+(?:by|via|with)|"
    r"count\s+(?:by|via|with)|based\s+on)\b[^.;\n]{0,100}$", re.I,
)
_STEP = re.compile(
    r"\b(?:initialize|initialise|set|update|increment|decrement)\s+"
    rf"(?:(?:the\s+)?(?:state|table|counter|accumulator|dp)\b|{_SYMBOL}\s*(?:=|to\b))"
    r"[^.;\n]{0,100}|"
    r"\b(?:iterate|loop|recur)\s+(?:over|through|from)\b[^.;\n]{0,100}", re.I,
)
_CODE = re.compile(
    r"```(?:python|py|javascript|js|code)?\s*\n.*?```|"
    r"\b(?:print|exec|eval)\s*\([^\n]{0,200}\)|"
    r"\bfor\s+\w+\s+in\s+[^\n;]{1,100}:|"
    r"\bdef\s+\w+\s*\([^\n]{0,100}\)\s*:", re.I | re.S,
)
_ROUTING = re.compile(
    r"\b(?:mace|runtime|model)\s*(?:router|routing|selection)\b|"
    r"\b(?:select|choose|rank|evaluate)\b.{0,24}\b(?:candidate\s+)?models?\b|"
    r"\b(?:route\s+(?:to|via)|use|select|choose)\s+"
    r"(?:Qwen[\w.\-]*|GPT[\w.\-]*|DeepSeek[\w.\-]*|Claude[\w.\-]*)\b|"
    r"\b(?:worker\s+)?model\s*(?:=|:)\s*\S+", re.I,
)
_PROCESS_LANGUAGE = re.compile(
    r"\b(?:use|apply|first|then|next)\b[^.;\n]{0,160}"
    r"\b(?:derive|calculate|compute|prove|solve|obtain)\b", re.I,
)


def canonical_math(text: str) -> str:
    """Normalize representation, preserving variable case and expression structure."""
    text = str(text).replace("\\(", "").replace("\\)", "").replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"&(?=\s*[=+\-])", "", text)  # Alignment, not a matrix column separator.
    text = re.sub(r"\bm(?=\\angle)", "", text)  # m∠ denotes the angle's measure.
    def vector(match):
        content = match.group(1)
        if '&' in content:
            return match.group()
        return '(' + ','.join(part.strip() for part in content.split(r'\\')) + ')'
    text = re.sub(r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}", vector, text, flags=re.S)
    text = re.sub(r"\\[,;:! ]", "", text)
    text = text.replace(r'\pi', 'pi').replace(r'\theta', 'θ')
    text = text.replace(r'\lceil', 'ceil(').replace(r'\rceil', ')')
    text = text.replace(r'\lfloor', 'floor(').replace(r'\rfloor', ')')
    text = re.sub(r"\\(sin|cos|tan|arctan|arcsin|arccos)\b", r"\1", text)
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    text = re.sub(r"_\{([A-Za-z0-9]+)\}", r"_\1", text)
    text = re.sub(r"\\(?:tfrac|dfrac|frac)\s+([A-Za-z0-9])\s*([A-Za-z0-9])\b", r"\\frac{\1}{\2}", text)
    for _ in range(16):
        changed = re.sub(r"\\(?:tfrac|dfrac|frac)\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", text)
        if changed == text:
            break
        text = changed
    for _ in range(16):
        changed = re.sub(r"\\(?:mathbf|overrightarrow)\{([^{}]*)\}", r"\1", text)
        if changed == text:
            break
        text = changed
    text = re.sub(r"\\(?:cdot|times)\s*", "*", text)
    text = re.sub(r"\\equiv\s*", "=", text)
    text = re.sub(r"\\pmod\{([^{}]+)\}", r"mod(\1)", text)
    text = re.sub(r"\(\s*mod\s+(\d+)\s*\)", r"mod(\1)", text, flags=re.I)
    text = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", text)
    text = re.sub(r"\^\(([A-Za-z0-9]+)\)", r"^\1", text)
    text = re.sub(r"\b(tan|sin|cos)\^\(-1\)(?=\s*\()", r"a\1", text)
    text = re.sub(r"\barc(tan|sin|cos)\b", r"a\1", text)
    text = re.sub(rf"\\log_(\d+)\s*({_SYMBOL})", r"log_\1(\2)", text)
    # Remove parentheses around atomic fraction operands only.
    text = re.sub(r"(?<![A-Za-z0-9_])\(([A-Za-z0-9]+(?:_[A-Za-z0-9]+)?)\)(?=/)", r"\1", text)
    text = re.sub(r"(?<=/)\(([A-Za-z0-9]+(?:_[A-Za-z0-9]+)?)\)", r"\1", text)
    # Innermost radicals first, so nested radicals keep their parentheses.
    for _ in range(16):
        changed = re.sub(r"\\sqrt\[(\d+)\]\{([^{}]*)\}", r"root_\1(\2)", text)
        changed = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", changed)
        if changed == text:
            break
        text = changed
    text = re.sub(r"\bsqrt\[(\d+)\]", r"root_\1", text)
    text = re.sub(rf"(?i:square\s+root\s+of)\s+(?:\$)?({_SYMBOL})(?:\$)?", r"sqrt(\1)", text)
    text = re.sub(rf"(?:\\angle|\bangle)\s+((?-i:{_SYMBOL}))", r"angle(\1)", text, flags=re.I)
    text = re.sub(r"\^\\circ\b|\bdegrees\b", "", text)
    text = re.sub(r"\b(?:equals|is equal to)\b", "=", text, flags=re.I)
    text = re.sub(r"\b(?:can be (?:written|expressed) as|is given by)\b", "=", text, flags=re.I)
    text = re.sub(r"\bmeasures\b", "=", text, flags=re.I)
    text = re.sub(r"\s*([=^_()+*/\-])\s*", r"\1", text)
    text = re.sub(r"\{\s*", "{", text)
    text = re.sub(r"\s*\}", "}", text)
    text = re.sub(r"\s*,\s*", ",", text)
    # Repeated application and f^k(x) are representations of the same iteration,
    # not a derived identity about the function's values.
    for _ in range(16):
        changed = re.sub(r'\b(?P<f>[A-Za-z]+)\((?P=f)(?:\^(?P<n>\d+))?\((?P<arg>[^()]*)\)\)',
                         lambda m: f"{m['f']}^{int(m['n'] or 1)+1}({m['arg']})", text)
        if changed == text:
            break
        text = changed
    return " ".join(text.split()).rstrip(".?")


def _formula_key(text: str) -> str:
    """Represent explicit/implicit multiplication without algebraic rewriting."""
    tokens = _MATH_TOKEN.findall(canonical_math(text))
    expanded = []
    for token in tokens:
        if (re.fullmatch(r'[a-z]{2,3}|[a-z][A-Z]', token)
                and token not in {'sin', 'cos', 'tan', 'atan', 'asin', 'acos', 'log', 'exp', 'ln', 'pi', 'dp', 'mod'}):
            expanded.extend(token)
        else:
            expanded.append(token)
    result = []
    functions = {'sqrt', 'sin', 'cos', 'tan', 'atan', 'asin', 'acos', 'ceil', 'floor', 'log', 'ln', 'exp', 'angle', 'f', 'g', 'h'}
    for token in expanded:
        if result:
            previous = result[-1]
            left_atom = previous in {')', '}', ']', '!'} or previous[0].isalnum()
            right_atom = token == '(' or token[0].isalnum()
            function = previous in functions or bool(re.fullmatch(r'(?:root|log)_\d+', previous))
            if left_atom and right_atom and not (token == '(' and function):
                result.append('*')
        result.append(token)
    return ''.join(result)


@dataclass(frozen=True)
class PublicTask:
    prose: str
    objects: tuple[dict, ...]
    goals: tuple[dict, ...]

    def evidence(self, fragment: str) -> dict | None:
        """Exact normalized fragment grounding, never mere shared-symbol grounding."""
        needle = canonical_math(fragment)
        if not needle:
            return None
        ratio = self.public_ratio(needle)
        if ratio:
            return ratio
        # Formula comparisons are case-sensitive. Prose/method names can casefold.
        formula = bool(re.search(r"[=^_]|sqrt\(", needle))
        if '=' in needle:
            for match in re.finditer(r'\$+([^$]+)\$+|\\\[(.*?)\\\]|\\\((.*?)\\\)', self.prose, re.S):
                expression = next(group for group in match.groups() if group is not None)
                if any(_formula_key(eq.group()) == _formula_key(needle) for eq in equalities(canonical_math(expression))):
                    return {"span": [match.start(), match.end()], "text": match.group()[:240]}
        for match in re.finditer(r"(?:[^.!?\n]|\.(?=\d))+[.!?]?", self.prose):
            candidate = canonical_math(match.group())
            if '=' in needle:
                # Compare the complete equality, never a suffix of its left side.
                if any(_formula_key(eq.group()) == _formula_key(needle) for eq in equalities(candidate)):
                    return {"span": [match.start(), match.end()], "text": match.group()[:240]}
                continue
            comparable = candidate if formula else candidate.casefold()
            target = needle if formula else needle.casefold()
            pos = comparable.find(target)
            while pos >= 0:
                end = pos + len(target)
                # x=2 must not match x=20 or 2*x=2, and N must not match n.
                if ((pos == 0 or comparable[pos - 1] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_*/+-^")
                        and (end == len(comparable) or comparable[end] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_*/+-^")):
                    return {"span": [match.start(), match.end()], "text": match.group()[:240]}
                pos = comparable.find(target, pos + 1)
        return None

    def public_ratio(self, fragment: str) -> dict | None:
        """Match a stated quantity ratio, including both ordered entities.

        This only changes notation: it does not infer ratios, simplify numbers,
        or ground an isolated suffix such as (AIC)=125:6.
        """
        quantity = r'perimeter|area|volume|length'
        entity = r'[A-Z]{1,4}'
        number = r'\d+(?:\.\d+)?'
        relation = re.fullmatch(
            rf'(?P<kind>(?i:{quantity}))\((?P<left>{entity})\)\s*[:/]\s*'
            rf'(?P<kind2>(?i:{quantity}))\((?P<right>{entity})\)\s*=\s*'
            rf'(?P<n>{number})\s*[:/]\s*(?P<d>{number})', fragment)
        if not relation or relation['kind'].lower() != relation['kind2'].lower():
            return None
        noun = relation['kind'].lower()
        # Normalize LaTeX labels without dropping the quantity or entity order.
        for sentence in re.finditer(r'(?:[^.!?\n]|\.(?=\d))+[.!?]?', self.prose):
            text = canonical_math(sentence.group())
            text = re.sub(r'\\triangle\s*', 'triangle ', text)
            pattern = (rf'\b(?i:{noun}s?)\s+(?i:of)\s+(?i:the\s+)?'
                rf'(?i:(?:triangle|segment|solid|circle|polygon)s?\s+)?{re.escape(relation["left"])}\s+'
                rf'(?i:and)\s+(?i:the\s+)?(?i:(?:triangle|segment|solid|circle|polygon)s?\s+)?'
                rf'{re.escape(relation["right"])}\s+(?i:(?:are\s+)?in\s+(?:the\s+)?ratio)\s+'
                rf'{re.escape(relation["n"])}\s*[:/]\s*{re.escape(relation["d"])}(?!\d|\.\d)')
            if re.search(pattern, text):
                return {'span': list(sentence.span()), 'text': sentence.group()[:240],
                        'quantity': noun, 'entities': [relation['left'], relation['right']]}
        return None

    def named_quantity(self, fragment: str) -> dict | None:
        """Ground explicit numeric height/radius/etc. constraints in public prose.

        A prose quantity tied to a public equation symbol can also have a stated
        target value. This copies the stated value; it never evaluates that equation.
        Counts are intentionally excluded: equal numbers of different objects do
        not establish that a path count equals a grid's number of squares.
        """
        match = re.fullmatch(rf'({_SYMBOL})\s*=\s*(\d+(?:\.\d+)?)', fragment)
        if not match:
            return None
        symbol, value = match.groups()
        quantities = {'height': r'height|raised', 'radius': r'radius',
                      'area': r'area', 'length': r'length', 'distance': r'distance'}
        for definition in re.finditer(r'\$+([^$]+)\$+', self.prose, re.S):
            if not re.match(re.escape(symbol) + r'\s*=', canonical_math(definition.group(1))):
                continue
            prefix = self.prose[max(0, definition.start()-150):definition.start()]
            for noun, aliases in quantities.items():
                if not re.search(r'\b'+noun+r'\b', prefix, re.I):
                    continue
                given = re.search(r'\b(?:'+aliases+r')\b[^.!?$]{0,70}?\b'+re.escape(value)+r'\b', self.prose, re.I)
                if given:
                    return {'span': list(given.span()), 'text': given.group(), 'quantity': noun, 'symbol': symbol}
        return None

    def given_alias(self, fragment: str) -> dict | None:
        """A fresh label for an explicit public input supplies no computed value."""
        parts = re.split(r'\s*(?:=|\b(?:equals|is equal to)\b)\s*', fragment, maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(_SYMBOL, parts[0]):
            return None
        lhs, rhs = parts
        if lhs in {'LCM', 'GCD'}:
            return None
        maths = list(re.finditer(r'\$+([^$]+)\$+|\\\[(.*?)\\\]|\\\((.*?)\\\)', self.prose, re.S))
        # Existing public objects cannot be assigned a different number/formula.
        symbols = {obj['symbol'] for obj in self.objects}
        for match in maths:
            symbols.update(re.findall(rf'(?<![\w]){_SYMBOL}(?![\w])', match.group()))
        base = lambda value: value.split('_', 1)[0].casefold()
        if lhs in symbols or any(lhs.casefold() == s.casefold() or base(lhs) == base(s) for s in symbols):
            return None
        key = _formula_key(rhs)
        for match in maths:
            expression = next(group for group in match.groups() if group is not None)
            if _formula_key(expression) == key:
                return {"span": [match.start(), match.end()], "text": match.group()[:240]}
        # Fresh index/data labels can restate public numeric inputs, not new numbers.
        if re.fullmatch(r'\d+', rhs) and re.search(r'(?<![\d.])' + re.escape(rhs) + r'(?![\d.])', self.prose):
            return {"source": "public_literal_fresh_label", "text": rhs}
        return None

    def summary(self) -> dict:
        return {"source": "public_task_only", "objects": list(self.objects[:6]),
                "goals": list(self.goals[:4])}


def analyze_public_task(public_task: str) -> PublicTask:
    # Keep original offsets but exclude diagram program statements as authority.
    prose = _DIAGRAM.sub(lambda m: " " * len(m.group()), str(public_task or ""))
    objects = []
    for m in re.finditer(rf"\b(?:let|denote|define)\s+\$?({_SYMBOL})\$?\s+(?:be|denote|=)\s+([^.!?\n]+)", prose, re.I):
        objects.append({"symbol": m.group(1), "definition": m.group(2)[:200], "span": [m.start(), m.end()]})
    goals = []
    for m in re.finditer(r"\b(?:find|determine|compute|calculate|evaluate|what\s+is)\s+([^.!?\n]+)", prose, re.I):
        goals.append({"text": m.group(1)[:200], "expression": canonical_math(m.group(1))[:200],
                      "span": [m.start(1), m.end(1)]})
    return PublicTask(prose, tuple(objects), tuple(goals))


def scan_boundaries(text: str, *, public: PublicTask, action_names: Iterable[str] = ()) -> list[dict]:
    """Find concrete prohibited fragments independently of lawful target fragments."""
    issues: list[dict] = []

    def add(code: str, rule: str, match: re.Match, reason: str, *, groundable: bool = True,
            span: tuple[int, int] | None = None) -> None:
        start, end = span or match.span()
        fragment = text[start:end]
        if groundable and public.evidence(fragment):
            return
        issues.append({"code": code, "rule_id": rule, "span": [start, end],
                       "matched_text": fragment[:180], "reason": reason,
                       "public_provenance": None,
                       "boundary": "Public objects and goals may be restated; new solution/control content may not."})

    for m in _ANSWER.finditer(text):
        public_abstention = re.search(r"\b(?:respond|reply)\s+with\s+['\"]?unanswerable\b", public.prose, re.I)
        conditional = re.search(r"\botherwise\b[^.!?]{0,100}$", text[:m.start()], re.I)
        if public_abstention and conditional and re.fullmatch(r"answer\s+is\s+['\"]?unanswerable['\"]?", m.group(), re.I):
            continue
        value = re.split(r"\b(?:is|equals)\b|[=:：]|是", m.group(), maxsplit=1, flags=re.I)[-1].strip()
        if not re.match(r"[+\-\d$\\]|sqrt\s*\(|['\"]?UNANSWERABLE\b|[A-Za-z](?:_[A-Za-z0-9]+)?\s*(?:$|[+*/^=(])", value, re.I):
            # 'answer is an exact value' describes a deliverable, not an answer.
            continue
        add("answer_or_solution_leak", "declared_answer", m,
            "Director declares an answer rather than asking the Worker to determine it.")
    for m in _VALUE_STATEMENT.finditer(text):
        add("answer_or_solution_leak", "supplied_value", m,
            "Director supplies a numerical result not grounded in the public task.")
    for m in equalities(text):
        if public.given_alias(m.group()) or public.named_quantity(m.group()):
            continue
        add("answer_or_solution_leak", "unpublished_assignment", m,
            "This value/equality is not supplied by the public task; a public symbol does not authorize a new value.")
    for m in _CODE.finditer(text):
        add("concrete_solution_procedure", "solution_code", m, "Director supplies executable solution code.")
    for m in _ALGORITHM.finditer(text):
        prefix_start = max(0, m.start() - 110)
        prefix = text[prefix_start:m.start()]
        suffix = text[m.end():m.end() + 70]
        directive = _METHOD_DIRECTIVE.search(prefix)
        context = re.match(r"\s+(?:over|with|on|to|using)\b[^.;\n]{0,60}", suffix, re.I)
        if directive or context:
            add("concrete_solution_procedure", "prescribed_algorithm", m,
                "Director selects a specific non-public algorithm for execution.",
                span=(prefix_start + directive.start(), m.end()) if directive
                else (m.start(), m.end() + context.end()))
    for m in _STEP.finditer(text):
        add("concrete_solution_procedure", "algorithm_step", m,
            "Director prescribes initialization/update/iteration steps absent from the public task.")
    for m in _ROUTING.finditer(text):
        add("runtime_routing_control", "model_route", m,
            "Runtime/model routing belongs to SET_MODEL, not Worker responsibility text.", groundable=False)
    for raw_name in sorted(set(action_names)):
        name = str(raw_name or "").strip()
        if not name:
            continue
        escaped = re.escape(name)
        pattern = (rf"\b{escaped}\b" if "_" in name else
                   rf"\b(?:call|invoke|execute|run)\b.{{0,24}}\b{escaped}\b|"
                   rf"\buse\b.{{0,16}}\b{escaped}\b(?!\s+(?:results?|output|evidence|findings|records?|documents?))|"
                   rf"\b(?:use|select|choose)\b.{{0,16}}\b{escaped}\b\s+(?:action|tool|function|api)\b|"
                   rf"\b{escaped}\b\s+(?:action|tool|function|api)\b")
        for m in re.finditer(pattern, text, re.I):
            add("worker_action_control", "worker_action", m,
                "Worker chooses its own Actions; remove the prescribed Action/tool.", groundable=False)
    # Stable textual order, bounded at the consumer after cross-field mapping.
    return sorted(issues, key=lambda item: (item["span"][0], item["rule_id"]))


def process_language_audit(text: str, public: PublicTask) -> list[dict]:
    """Audit weak lexical cues without promoting them to hard violations."""
    return [{"rule_id": "process_language_only", "span": [m.start(), m.end()],
             "matched_text": m.group()[:180], "decision": "not_a_hard_violation",
             "public_provenance": public.evidence(m.group())}
            for m in list(_PROCESS_LANGUAGE.finditer(text))[:4]]


def _normalized_offset(text: str, offset: int) -> int:
    """Map an original boundary into the existing whitespace-only normalization."""
    length = 0
    for token in re.finditer(r"\S+", text):
        if length:
            length += 1
        if offset <= token.start():
            return length
        if offset <= token.end():
            return length + offset - token.start()
        length += token.end() - token.start()
    return length


def validate_fields(fields: dict[str, str], *, public_task: str, action_names: Iterable[str]) -> tuple[list[dict], dict]:
    public = analyze_public_task(public_task)
    issues = []
    audit = {"public_task": {**public.summary(), "diagram_code_excluded": bool(_DIAGRAM.search(public_task))}, "process_language": []}
    segments = []
    joined = ""
    action_names = tuple(action_names)
    for name, text in fields.items():
        if joined:
            joined += " "
        segments.append((name, len(joined), len(joined) + len(text)))
        joined += text
        audit["process_language"].extend({"field": name, **item} for item in process_language_audit(text, public))
        for item in scan_boundaries(text, public=public, action_names=action_names):
            start, end = item["span"]
            issues.append({**item, "field": name, "original_span": [start, end],
                           "normalized_span": [_normalized_offset(text, start), _normalized_offset(text, end)]})
    # Joining complete original fields detects split declarations/algorithm names.
    # Only add matches that actually cross a field boundary, avoiding duplicates.
    for item in scan_boundaries(joined, public=public, action_names=action_names):
        start, end = item["span"]
        locations = [{"field": name, "span": [max(start, a) - a, min(end, b) - a]}
                     for name, a, b in segments if a < end and b > start]
        if len(locations) > 1:
            if item['rule_id'] == 'unpublished_assignment':
                first_name, boundary, boundary_end = next(segment for segment in segments if segment[0] == locations[0]['field'])
                before, after = joined[start:boundary_end].rstrip(), joined[boundary_end:end].lstrip()
                continuation = (before.endswith(('=', '+', '-', '*', '/', '^'))
                                or after.startswith(('+', '-', '*', '/', '^'))
                                or any(before.count(a) > before.count(b) for a, b in [('(', ')'), ('[', ']'), ('{', '}')]))
                if not continuation:
                    continue  # A completed expression does not multiply the next field's prose.
            issues.append({**item, "field": "+".join(loc["field"] for loc in locations),
                           "field_locations": locations, "original_span": [start, end],
                           "normalized_span": [_normalized_offset(joined, start), _normalized_offset(joined, end)]})
    audit["process_language"] = audit["process_language"][:8]
    return issues, audit
