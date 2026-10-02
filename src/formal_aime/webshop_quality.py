"""Public-requirement scoring for the project's versioned WebShop release.

Standard-library only: this module also runs in the pinned simulator Python.
The upstream scorer remains available as a diagnostic. No ASIN equality bonus,
hidden-title similarity, fuzzy numeric matching, or answer access by the agent.
"""

from __future__ import annotations
import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence

VERSION = "webshop-quality-20260930-v1"


def public_priced_product(product, price):
    """The simulator's actual fixed purchase price is public before purchase."""
    value = dict(product)
    value.update(Price="$" + str(float(price)), pricing=[float(price)])
    return value


class PricedCatalog(Mapping):
    def __init__(self, catalog, prices):
        self.catalog = catalog
        self.prices = prices

    def __len__(self):
        return len(self.catalog)

    def __iter__(self):
        return iter(self.catalog)

    def __getitem__(self, key):
        return public_priced_product(self.catalog[key], self.prices[key])


class PricedProducts(Sequence):
    def __init__(self, products, prices):
        self.products = products
        self.prices = prices

    def __len__(self):
        return len(self.products)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        product = self.products[index]
        return public_priced_product(product, self.prices[product["asin"]])


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = text.translate(
        str.maketrans(
            {"’": "'", "‘": "'", "“": '"', "”": '"', "‐": "-", "‑": "-", "–": "-", "—": "-"}
        )
    )
    for word, number in [
        ("twenty-four", "24"),
        ("sixteen", "16"),
        ("eleven", "11"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
        ("ten", "10"),
        ("twelve", "12"),
    ]:
        text = re.sub(r"\b" + word + r"\b", number, text)
    text = re.sub(r"(?<!\d)\.(\d)", r"0.\1", text)
    text = re.sub(r'(?<=\d)\s*"', " inch ", text)
    text = re.sub(r"(?<=\d)\s*\x27", " ft ", text)
    text = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text)
    text = re.sub(r"\bextra[ -]*extra[ -]*large\b|\bxx[ -]*large\b", "xxl", text)
    text = re.sub(r"\bextra[ -]*large\b|\bx[ -]*large\b", "xl", text)
    text = re.sub(r"\bextra[ -]*small\b|\bx[ -]*small\b", "xs", text)
    text = text.replace("+", " plus ")
    text = re.sub(r"\b(\d)\s*x[ -]*large\b", r"\1xl", text)
    text = re.sub(r"\b(\d)\s*x[ -]*small\b", r"\1xs", text)
    for pattern, replacement in [
        (r"\bgrey\b", "gray"),
        (r"\binches\b", "inch"),
        (r"\bfeet\b|\bfoot\b", "ft"),
        (r"\bounces?\b", "oz"),
        (r"\bpounds?\b", "lb"),
        (r"\bcentimet(?:er|re)s?\b", "cm"),
        (r"\bmillimet(?:er|re)s?\b", "mm"),
        (r"\bfluid\s+oz\b", "fl oz"),
        (r"\bfl\.", "fl"),
        (r"\b(?:pair|pairs)\b", "pair"),
        (r"\b(?:pcs|pieces)\b", "piece"),
        (r"\bpacks\b", "pack"),
        (r"\b(?:gigabytes?|gigs?)\b", "gb"),
        (r"\bterabytes?\b", "tb"),
        (r"\bmemory\b", "ram"),
        (r"\bset\s+of\s+(\d+)\b", r"pack \1"),
        (r"\bpack\s+of\s+(\d+)\b", r"pack \1"),
        (r"\b(\d+)\s*-?\s*pack\b", r"pack \1"),
    ]:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\b(\d+)\.0+\b", r"\1", text)
    text = re.sub(r"[^\w.]+", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    return " ".join(text.split())


def tokens(text):
    return normalize(text).split()


def equivalent(left, right):
    """Order-insensitive words, but numbers, negation and multiplicity survive."""
    from collections import Counter

    def quantities(value):
        text = normalize(value)
        measures = re.findall(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s+(oz|lb|ft|inch|cm|mm|gb|tb|count|women|men)\b", text
        )
        packs = re.findall(r"\bpack\s+(\d+)\b", text)
        dimensions = re.findall(r"\d+(?:\.\d+)?(?:\s+(?:inch\s+)?x\s+\d+(?:\.\d+)?)+", text)
        return sorted(measures), sorted(packs), dimensions

    if quantities(left) != quantities(right):
        return False

    def parts(s):
        value = normalize(s)
        value = re.sub(r"\b(?:with|the|of|for)\b", "", value)
        return Counter(value.split())

    return parts(left) == parts(right)


def group_name(group):
    name = normalize(group)
    return {
        "colour": "color",
        "pattern name": "pattern",
        "flavor name": "flavor",
        "scent name": "scent",
        "style name": "style",
        "processor description": "processor",
        "memory size": "ram",
        "digital storage capacity": "storage",
        "hard disk size": "storage",
        "number of items": "quantity",
        "item package quantity": "quantity",
    }.get(name, name)


def _contains(needle, haystack):
    return bool(
        re.search(r"(?<![\w.])" + re.escape(normalize(needle)) + r"(?![\w.])", normalize(haystack))
    )


ATTRIBUTE_ALIASES = {
    "easy carry": ["easy to carry", "easy carrying", "portable"],
    "light weight": ["lightweight"],
    "easy clean": ["easy to clean", "easy cleaning"],
    "easy use": ["easy to use", "simple to use"],
    "easy apply": ["easy to apply", "easy application"],
    "easy install": ["easy to install", "easy installation", "easily installation"],
    "machine washable": ["machine wash", "machine wash cold"],
    "machine wash": ["machine washable"],
    "long handle": ["long wooden handle", "long handled"],
    "assembly required": ["requires assembly", "leg assembly is required"],
    "anti slip": ["non-slip", "slip resistant", "non-skid"],
    "slip resistant": ["non-slip", "non-skid"],
    "rubber sole": ["rubber outsole", "rubber outsoles"],
    "rubber outsole": ["rubber sole", "rubber outsoles"],
    "usb port": ["usb ports", "usb-c port", "usb charging port"],
    "high definition": ["hd", "1080p", "full hd", "720p"],
    "ultra hd": ["4k", "8k"],
    "short sleeve": ["short sleeves", "short sleeved"],
    "long sleeve": ["long sleeves", "long sleeved"],
    "double sided": ["double-sided"],
    "lace closure": ["lace-up closure", "lace up"],
    "fresh breath": ["freshens breath", "fresh breath", "breath freshening", "freshen breath"],
    "teeth whitening": ["tooth whitening", "whitens teeth", "whitening toothpaste"],
    "anti aging": ["anti-ageing", "anti-wrinkle", "fine lines", "wrinkles"],
    "non toxic": ["nontoxic"],
    "non gmo": ["non-gmo"],
    "coated steel": ["steel coated"],
    "fleece throw": ["fleece blanket", "throw blanket"],
    "wall mounted": ["wall-mounted", "wall mount", "wall sconce", "wall lamp"],
    "queen size": ["queen"],
    "king size": ["king"],
    "white item": ["white"],
    "compatible apple": ["compatible with apple", "for apple watch", "for iwatch"],
    "noise cancelling": ["noise cancellation", "noise reduction"],
    "high quality": ["premium quality"],
    "super soft": ["ultra soft", "ultra-soft"],
    "comfortable fit": ["comfortable", "comfort"],
    "vinyl acetate": ["ethylene vinyl acetate", "eva"],
    "ethylene vinyl": ["ethylene vinyl acetate", "eva"],
    "ready eat": ["ready to eat"],
    "ready use": ["ready to use"],
    "easy prepare": ["easy to prepare"],
    "plug play": ["plug and play", "plug-and-play"],
    "long lasting": ["long-lasting", "long wear", "long wearing"],
    "natural ingredients": ["all natural", "100% natural"],
    "high waist": ["high waisted"],
    "butt lifting": ["butt lift", "butt-lifting"],
    "elastic waist": ["elastic waistband"],
    "ready hang": ["ready to hang", "easy to hang"],
    "metal legs": ["steel legs", "chrome legs", "hairpin legs"],
    "quality polyester": ["polyester"],
    "quality materials": ["high quality material", "premium material"],
    "low carb": ["low carbohydrate", "zero carb", "zero carbs"],
    "trader joe": ["trader joe's"],
    "hand crafted": ["handcrafted", "hand made", "handmade"],
    "great gift": ["gift"],
    "perfect gift": ["gift"],
    "easy assemble": ["easy to assemble", "easy assembly"],
    "height adjustable": [
        "adjustable height",
        "adjustable seat height",
        "height can be adjusted",
        "height adjustment",
    ],
    "1080p hd": ["1080p full hd"],
    "drawstring closure": ["drawstring"],
    "non slip": ["non-slip", "slip resistant", "non-skid", "anti slip", "no-skid"],
    "button closure": ["button down", "buttons", "button placket"],
    "wood finish": ["wood design", "wooden finish", "wood with a finish", "acacia wood"],
    "power amplifier": ["power amp", "amplifier"],
    "low fat": ["low in fat"],
    "everyday wear": ["everyday look", "daily wear"],
    "cake topper": ["cupcake topper"],
    "coaxial cable": ["coax cable", "coaxial rg-6", "rg-6 coaxial"],
}


def _sentences(product):
    # Never use the extracted Attributes labels as evidence for themselves.
    sources = [
        str(product.get("Title", "")),
        *product.get("BulletPoints", []),
        str(product.get("Description", "")),
    ]
    out = []
    for source in sources:
        for sentence in re.split(r"[!?;\n]|(?<!\d)\.(?!\d)|<br\s*/?>", source):
            if sentence.strip():
                out.append(sentence)
    return out


def attribute_match(product, requirement):
    phrase = requirement["phrase"]
    polarity = requirement.get("polarity", True)
    aliases = [phrase, *ATTRIBUTE_ALIASES.get(phrase, []), *requirement.get("aliases", [])]
    # Convert intrinsically negative formulations into a positive subject and
    # a negative polarity; the same predicate is then used for all products.
    negative = re.fullmatch(r"(.+) free", phrase)
    if negative and negative[1] not in ("cruelty",):
        aliases = [negative[1]]
        polarity = False
    if phrase == "non alcoholic":
        aliases = ["alcohol", "alcoholic"]
        polarity = False
    if phrase == "zero sugar":
        aliases = ["sugar"]
        polarity = False
    if phrase == "non dairy":
        aliases = ["dairy"]
        polarity = False
    matches = []
    for original in _sentences(product):
        sentence = normalize(original)
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(normalize(alias)) + r"(?!\w)"
            for hit in re.finditer(pattern, sentence):
                before = sentence[: hit.start()].split()[-6:]
                after = sentence[hit.end() :].split()[:5]
                prefix = " ".join(before)
                neg = bool(
                    re.search(
                        r"\b(?:no|not|non|without|zero|0 g|free from|free of)(?: a| the| any| added| containing| contain| include| includes)?$",
                        prefix,
                    )
                )
                neg = neg or bool(after and after[0] == "free")
                # A coordinated allergen list retains the leading negation.
                # Comma-separated ingredient lists retain negation until a new
                # clause. Work in the normalized coordinates of this hit.
                entire_prefix = sentence[: hit.start()]
                if not polarity and re.search(
                    r"\b(?:no|without|does not contain|free from|free of|instead of)\b(?:(?!\b(?:but|which|our|is|are|we|has|have)\b).){0,70}$",
                    entire_prefix,
                ):
                    neg = True
                if re.search(r"(?:don t|doesn t) (?:put|include|contain)(?: any)?$", prefix):
                    neg = True
                if re.search(
                    r"\b(?:tastes? (?:just )?like|similar in taste to|compared (?:with|to)|removal of)\b",
                    entire_prefix[-75:],
                ):
                    continue
                # "batteries not included" negates inclusion, not existence.
                if phrase in ("batteries included", "aaa batteries") and "not included" in sentence:
                    neg = True
                if phrase == "sensitive skin" and re.search(
                    r"(?:do not|avoid).{0,35}sensitive skin", sentence
                ):
                    neg = True
                if not polarity and not neg:
                    # Mentioning an ingredient in a comparison or recipe is
                    # not an assertion that the sold product contains it.
                    affirmative = re.search(
                        r"\b(?:contains?|includes?|with|added|ingredients?|made (?:with|from))\b(?:(?!\b(?:no|without|free|instead)\b).){0,45}$",
                        prefix,
                    )
                    if not affirmative:
                        continue
                matches.append({"text": original, "positive": not neg})
    if not matches:
        return False, {"status": "no_catalog_evidence", "phrase": phrase}
    if any(m["positive"] != polarity for m in matches):
        # A conflicting care/compatibility statement must not be ignored just
        # because a keyword also occurs elsewhere in a long advertising block.
        return False, {
            "status": "contradictory_evidence",
            "phrase": phrase,
            "evidence": matches[:5],
        }
    return True, {"status": "supported", "phrase": phrase, "evidence": matches[:3]}


def value_match(value, requirement):
    """Match only the public facets of an option, retaining numeric boundaries."""
    mode = requirement.get("match", "exact")
    allowed = requirement["values"]
    if mode == "contains":
        return any(_contains(x, value) for x in allowed)
    if mode == "all":
        return all(_contains(x, value) for x in allowed)
    if mode == "patterns":
        return all(re.search(x, normalize(value)) for x in allowed)

    def without_metric_note(text):
        return re.sub(r"\([\d\s.x×/]+(?:cm|mm|m|inch|inches)\)", "", text, flags=re.I).strip()

    return any(
        equivalent(value, x) or equivalent(without_metric_note(value), without_metric_note(x))
        for x in allowed
    )


def option_match(product, selected, requirement):
    group = group_name(requirement["group"])
    allowed = requirement["values"]
    groups = {group, *[group_name(g) for g in requirement.get("also_check_groups", [])]}
    relevant = [(k, v) for k, v in selected.items() if group_name(k) in groups]

    def match(value):
        return value_match(value, requirement)

    if relevant:
        ok = all(match(value) for _, value in relevant)
        return ok, {
            "status": "selected_match" if ok else "selected_conflict",
            "group": group,
            "values": [x[1] for x in relevant],
        }
    available = [k for k in product.get("options", {}) if group_name(k) in groups]
    if available:
        return False, {"status": "required_option_not_selected", "group": group}
    # Fixed-SKU products need no nonexistent click. Only the public catalog's
    # title/specifications may supply the fixed value; do not infer a default
    # from another product or from the requested goal itself.
    evidence = " ".join([str(product.get("Title", "")), *product.get("BulletPoints", [])])
    title_tokens = normalize(evidence)
    supported = value_match(
        title_tokens,
        {**requirement, "match": "contains"}
        if requirement.get("match", "exact") == "exact"
        else requirement,
    )
    aliases = {
        ("style", "tea bags"): ["sachets", "tea bags"],
        ("flavor", "peppermint leaf"): ["peppermint tea", "whole leaf peppermint"],
    }
    if not supported:
        supported = any(
            _contains(alias, evidence)
            for expected in allowed
            for alias in aliases.get((group, normalize(expected)), [])
        )
    return supported, {
        "status": "fixed_catalog_match" if supported else "fixed_value_unconfirmed",
        "group": group,
    }


def type_match(product, contract, options=None):
    patterns = contract["type_patterns"]
    title = normalize(product.get("Title", ""))
    # Ignore compatibility targets after "for" when the sold object is a
    # power cable, cover, charger, or another accessory.
    accessory = bool(
        re.search(r"\b(?:power cord|charging cable|charger|replacement cable)\b", title)
    )
    if accessory:
        title = re.split(r"\b(?:for|compatible)\b", title, maxsplit=1)[0]
    # Catalog variants sometimes represent different product forms (shampoo /
    # conditioner, sheets / duvet). Only selected variant values can override
    # the default title, never an unselected possible variant.
    variant_text = " ".join(normalize(v) for v in (options or {}).values())
    return any(
        re.search(pattern, title) or re.search(pattern, variant_text) for pattern in patterns
    )


def evaluate(product, goal, price, options):
    contract = goal["_quality_contract"]
    if contract["version"] != VERSION:
        raise ValueError("unknown WebShop quality contract")
    if hashlib.sha256(goal["instruction_text"].encode()).hexdigest() != contract["prompt_sha256"]:
        raise ValueError("WebShop prompt and scoring contract differ")
    for group, value in options.items():
        if group not in product.get("options", {}) or value not in product["options"][group]:
            raise ValueError("selected option is absent from the actual product")
    checks = []
    for req in contract["attributes"]:
        ok, detail = attribute_match(product, req)
        checks.append({"kind": "attribute", "matched": ok, **detail})
    for req in contract["options"]:
        ok, detail = option_match(product, options, req)
        checks.append({"kind": "option", "matched": ok, **detail})
    affordable = float(price) < float(goal["price_upper"])
    checks.append({"kind": "price", "matched": affordable, "status": "strict_upper_bound"})
    correct_type = type_match(product, contract, options)
    reward = (sum(c["matched"] for c in checks) / len(checks)) if correct_type else 0.0
    attrs = [x for x in checks if x["kind"] == "attribute"]
    opts = [x for x in checks if x["kind"] == "option"]
    parts = {
        "r_type": float(correct_type),
        "r_att": sum(x["matched"] for x in attrs) / len(attrs) if attrs else 1.0,
        "r_option": sum(x["matched"] for x in opts) / len(opts) if opts else 1.0,
        "r_price": affordable,
        "scorer_version": VERSION,
        "quality_checks": checks,
    }
    return reward, parts


def install(env_module):
    original = env_module.get_reward
    if getattr(original, "_quality_installed", False):
        return

    def get_reward(product, goal, price, options, **kwargs):
        if "_quality_contract" not in goal:
            return original(product, goal, price, options, **kwargs)
        reward, parts = evaluate(product, goal, price, options)
        # Preserve the untouched upstream result against the original labels.
        raw_goal = goal.get("_official_goal", goal)
        raw_score = original(product, raw_goal, price, options)
        parts["official_reward"] = raw_score
        return (reward, parts) if kwargs.get("verbose") else reward

    get_reward._quality_installed = True
    env_module.get_reward = get_reward
