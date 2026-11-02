"""Decides whether the shipment table can answer a question exactly, and answers it.

Plain rules, no LLM: no quota, the same result every time, and easy to test. The router only claims
a question when it understands every word of it. It removes what it recognises (IDs, product codes,
statuses, months, known destinations) and if any unknown word is left ("why", "Q3", "Tokyo") the
question goes to the document path. Without that, "How many shipments went to Tokyo?" would ignore
Tokyo and answer 10. A missed question still gets a document answer; a wrong claim gets a wrong one.
"""

import re
from dataclasses import dataclass, field

from shipment_rag.structured.store import ShipmentStore

ROUTE_DOCUMENTS = "documents"
ROUTE_TABLE = "shipment_table"

_ID = re.compile(r"\bsh-\d{4}-\d{3}\b")
_PRODUCT = re.compile(r"\btl-[a-z]+-\d+\b")
_MONTHS = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr", "may": "may",
    "june": "jun", "july": "jul", "august": "aug", "september": "sep", "october": "oct",
    "november": "nov", "december": "dec",
}
_MONTH_IN = re.compile(rf"\bin ({'|'.join(_MONTHS)})\b")
_STATUS_WORDS = {"delivered": "DELIVERED", "quarantined": "QUARANTINED",
                 "rejected": "REJECTED", "delayed": "DELAYED"}
_STATUS = re.compile(rf"\b({'|'.join(_STATUS_WORDS)})\b")
_NO_BREACH = re.compile(r"\b(?:no|without(?: a| any)?)(?: temperature)? (?:breach|breaches)\b")
_BREACH = re.compile(r"\b(?:breach|breaches|breached)\b")
_RATE = re.compile(r"\bbreach rate\b")
_GENERIC_PLACE_WORDS = {"hospital", "research", "camp", "office", "district", "health", "relief",
                        "rural", "distributor", "hub", "lab"}

_FILLER = {
    "a", "an", "the", "of", "in", "on", "for", "to", "was", "were", "is", "are", "did", "do",
    "does", "we", "have", "has", "had", "there", "any", "our", "that", "with", "by", "at", "and",
    "it", "its", "please", "me", "what", "which", "how", "many", "much", "number", "count",
    "total", "overall", "show", "list", "give", "all", "fy2024", "2024", "log", "this", "year",
    "shipment", "shipments", "consignment", "consignments", "sent", "shipped", "went", "go",
    "made", "record", "records", "recorded", "experienced", "temperature",
}
_UNITS = {"vials", "pouches", "boxes", "packs", "units"}
_HOT = {"highest", "maximum", "warmest", "hottest"}
_COLD = {"lowest", "minimum", "coldest"}
_FIELD_WORDS = {
    "status": "status", "state": "status", "product": "product", "batch": "batch",
    "quantity": "quantity", "vials": "quantity", "pouches": "quantity", "boxes": "quantity",
    "packs": "quantity", "route": "route", "destination": "route", "where": "route",
    "going": "route", "when": "dates", "date": "dates", "dates": "dates",
    "temperatures": "temperature", "temp": "temperature",
    "accepted": "accepted_by", "accept": "accepted_by", "who": "accepted_by",
    "received": "accepted_by", "delay": "delay", "late": "delay",
}
_MONTH_NAMES = {v: k.capitalize() for k, v in _MONTHS.items()}


@dataclass(frozen=True)
class StructuredQuery:
    template: str
    params: dict = field(default_factory=dict)


@dataclass
class StructuredResult:
    answer: str
    query: StructuredQuery
    rows: list[dict]
    refused: bool = False


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text))


def _plural(n: int, word: str = "shipment") -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


class StructuredRouter:
    def __init__(self, store: ShipmentStore):
        self.store = store
        self._aliases = self._destination_aliases()

    def _destination_aliases(self) -> dict[str, str]:
        """'bangalore' and 'genpath lab bangalore' both name that destination."""
        aliases = {}
        for row in self.store.rows():
            name = row["destination"].lower()
            aliases[name] = name
            for word in name.split():
                if len(word) >= 4 and word not in _GENERIC_PLACE_WORDS:
                    aliases.setdefault(word, word)
        return aliases

    def route(self, question: str) -> StructuredQuery | None:
        """The query that answers this exactly, or None to use the documents."""
        text = re.sub(r"[?!.,;:]+(\s|$)", " ", question.lower())
        ids = _ID.findall(text)
        if ids:
            return self._route_lookup(ids, _ID.sub(" ", text))
        return self._route_table(text)

    @staticmethod
    def _route_lookup(ids: list[str], rest: str) -> StructuredQuery | None:
        if len(set(ids)) != 1:
            return None
        chosen, unknown = [], set()
        rest = _STATUS.sub(" status ", rest)
        rest = _BREACH.sub(" temperature ", rest)
        for word in _words(rest) - _FILLER:
            if word in _FIELD_WORDS:
                chosen.append(_FIELD_WORDS[word])
            else:
                unknown.add(word)
        if "temperature" in _words(rest):
            chosen.append("temperature")
        if unknown or not chosen:
            return None
        return StructuredQuery("shipment_fields", {"shipment_id": ids[0].upper(),
                                                   "fields": sorted(set(chosen))})

    def _route_table(self, text: str) -> StructuredQuery | None:
        if _RATE.search(text):
            return self._only(_RATE.sub(" ", text), "breach_rate", {})

        filters = {}
        rest = text
        if product := _PRODUCT.search(rest):
            filters["product"] = product[0].upper()
            rest = _PRODUCT.sub(" ", rest)
        if month := _MONTH_IN.search(rest):
            filters["month"] = _MONTHS[month[1]]
            rest = _MONTH_IN.sub(" ", rest)
        if status := _STATUS.findall(rest):
            if len(set(status)) > 1:
                return None
            filters["status"] = _STATUS_WORDS[status[0]]
            rest = _STATUS.sub(" ", rest)
        if _NO_BREACH.search(rest):
            filters["breach"] = 0
            rest = _NO_BREACH.sub(" ", rest)
        elif _BREACH.search(rest):
            filters["breach"] = 1
            rest = _BREACH.sub(" ", rest)
        for alias in sorted(self._aliases, key=len, reverse=True):
            pattern = rf"\bto {re.escape(alias)}\b"
            if re.search(pattern, rest):
                filters["destination"] = alias
                rest = re.sub(pattern, " ", rest)
                break
        if accepted := re.search(r"\baccepted by ((?:dr )?[a-z]+(?: [a-z]+)?)\b", rest):
            # filter on the name as stored ("Dr. Anita Rao"), or a near miss would count 0
            known = {r["accepted_by"] for r in self.store.rows() if r["accepted_by"]}
            matches = [n for n in known if accepted[1] in n.lower().replace(".", "")]
            if len(accepted[1]) >= 4 and len(matches) == 1:
                filters["accepted_by"] = matches[0]
                rest = rest.replace(accepted[0], " ")

        words = _words(rest)
        about_shipments = bool(words & {"shipment", "shipments", "consignment", "consignments"})
        if words & _HOT and "temperature" in words:
            return None if filters else self._only(rest, "hottest", {}, _HOT)
        if words & _COLD and "temperature" in words:
            return None if filters else self._only(rest, "coldest", {}, _COLD)
        if words & {"status", "breakdown"} and about_shipments and not filters:
            return self._only(rest, "by_status", {}, {"status", "breakdown", "each", "per", "by"})
        if words & (_UNITS | {"quantity"}):
            return self._only(rest, "total_quantity", {"filters": filters}, _UNITS | {"quantity"})
        counting = "how many" in text or "number of" in text or "count of" in text
        if counting and (about_shipments or filters.get("breach") == 1):
            return self._only(rest, "count", {"filters": filters})
        listing = about_shipments and words & {"which", "list", "show", "what"}
        if listing and (filters or "all" in words):
            return self._only(rest, "list", {"filters": filters})
        return None

    @staticmethod
    def _only(rest: str, template: str, params: dict, extra: set | None = None
              ) -> StructuredQuery | None:
        """The claim stands only if every remaining word is understood."""
        if _words(rest) - _FILLER - (extra or set()):
            return None
        return StructuredQuery(template, params)

    def run(self, query: StructuredQuery) -> StructuredResult:
        return getattr(self, f"_run_{query.template}")(query)

    def answer(self, question: str) -> StructuredResult | None:
        query = self.route(question)
        return self.run(query) if query else None

    # ---------------- templates ----------------

    @staticmethod
    def _describe(filters: dict) -> str:
        parts = []
        if "status" in filters:
            parts.append(filters["status"].lower())
        if "product" in filters:
            parts.append(f"of {filters['product']}")
        if "destination" in filters:
            parts.append(f"to {filters['destination'].title()}")
        if "month" in filters:
            parts.append(f"in {_MONTH_NAMES[filters['month']]}")
        if "accepted_by" in filters:
            parts.append(f"accepted by {filters['accepted_by'].title()}")
        if filters.get("breach") == 1:
            parts.append("with a temperature breach")
        elif filters.get("breach") == 0:
            parts.append("with no temperature breach")
        return " ".join(parts)

    def _run_count(self, q: StructuredQuery) -> StructuredResult:
        filters = q.params["filters"]
        n, rows = self.store.count(filters), self.store.rows(filters)
        desc = self._describe(filters)
        ids = f": {', '.join(r['shipment_id'] for r in rows)}" if filters and rows else ""
        sentence = " ".join(filter(None, [_plural(n), desc, "in the FY2024 shipment log"]))
        return StructuredResult(f"{sentence}{ids}.", q, rows)

    def _run_list(self, q: StructuredQuery) -> StructuredResult:
        filters = q.params["filters"]
        rows = self.store.rows(filters)
        desc = self._describe(filters)
        if not rows:
            sentence = " ".join(filter(None, ["No shipments", desc, "in the FY2024 shipment log"]))
            return StructuredResult(f"{sentence}.", q, rows)
        items = ", ".join(f"{r['shipment_id']} ({r['product']} to {r['destination']})"
                          for r in rows)
        sentence = " ".join(filter(None, [_plural(len(rows)), desc]))
        return StructuredResult(f"{sentence}: {items}.", q, rows)

    def _run_total_quantity(self, q: StructuredQuery) -> StructuredResult:
        filters = q.params["filters"]
        totals = self.store.total_quantity(filters)
        rows = self.store.rows(filters)
        desc = self._describe(filters)
        if not totals:
            sentence = " ".join(filter(None, ["No shipments", desc, "in the FY2024 shipment log"]))
            return StructuredResult(f"{sentence}.", q, rows)
        parts = ", ".join(f"{t['total']} {t['unit']}" for t in totals)
        sentence = " ".join(filter(None, ["Total quantity shipped", desc]))
        return StructuredResult(f"{sentence}: {parts} across {_plural(len(rows))}.", q, rows)

    def _run_by_status(self, q: StructuredQuery) -> StructuredResult:
        groups = self.store.by_status()
        parts = ", ".join(f"{g['status'].lower()} {g['shipments']}" for g in groups)
        return StructuredResult(
            f"{_plural(sum(g['shipments'] for g in groups))} by status: {parts}.", q, groups)

    def _run_breach_rate(self, q: StructuredQuery) -> StructuredResult:
        total = self.store.count()
        rows = self.store.rows({"breach": 1})
        ids = ", ".join(r["shipment_id"] for r in rows)
        rate = len(rows) / total if total else 0.0
        return StructuredResult(
            f"The FY2024 temperature breach rate is {rate:.0%}: {len(rows)} of "
            f"{_plural(total)} ({ids}).", q, rows)

    @staticmethod
    def _temperature(row: dict) -> str:
        if row["temp_ambient"]:
            return f"{row['temp_max']:g}°C ambient"
        return f"{row['temp_min']:g}°C min / {row['temp_max']:g}°C max"

    def _run_hottest(self, q: StructuredQuery) -> StructuredResult:
        rows = self.store.hottest()
        items = ", ".join(f"{r['shipment_id']} ({self._temperature(r)})" for r in rows)
        return StructuredResult(
            f"The highest temperature recorded is {rows[0]['temp_max']:g}°C: {items}.", q, rows)

    def _run_coldest(self, q: StructuredQuery) -> StructuredResult:
        rows = self.store.coldest()
        items = ", ".join(f"{r['shipment_id']} ({self._temperature(r)})" for r in rows)
        return StructuredResult(
            f"The lowest temperature recorded is {rows[0]['temp_min']:g}°C: {items}.", q, rows)

    def _run_shipment_fields(self, q: StructuredQuery) -> StructuredResult:
        shipment_id = q.params["shipment_id"]
        row = self.store.get(shipment_id)
        if row is None:
            return StructuredResult(
                f"There is no shipment {shipment_id} in the FY2024 shipment log.", q, [],
                refused=True)
        return StructuredResult(
            f"{shipment_id}: " + "; ".join(self._field_text(name, row)
                                           for name in q.params["fields"]) + ".", q, [row])

    def _field_text(self, name: str, row: dict) -> str:
        if name == "status":
            note = f" ({row['status_note']})" if row["status_note"] else ""
            return f"status {row['status']}{note}"
        if name == "product":
            return f"product {row['product']}"
        if name == "batch":
            return f"batch {row['batch']}"
        if name == "quantity":
            return f"quantity {row['quantity']} {row['unit']}"
        if name == "route":
            return f"route {row['origin']} to {row['destination']}"
        if name == "dates":
            return f"dates {row['dates']}"
        if name == "temperature":
            breach = f", breach: {row['breach_detail']}" if row["breach"] else ", no breach"
            return f"temperature {self._temperature(row)}{breach}"
        if name == "accepted_by":
            return f"accepted by {row['accepted_by']}" if row["accepted_by"] else (
                "no acceptance recorded")
        if name == "delay":
            return f"delay {row['delay_days']} days" if row["delay_days"] else "no delay recorded"
        raise ValueError(f"unknown field: {name}")
