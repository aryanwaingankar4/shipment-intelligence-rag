"""The shipment log as a read-only SQLite table, queried only through fixed templates.

Nothing here runs SQL written by a user or a model. Every statement is a constant in this file and
every value is a bound parameter. The connection also refuses writes twice over: SQLite's
`query_only` mode, plus an authorizer that denies everything except reading the table.
"""

import re
import sqlite3
import threading
from dataclasses import astuple, dataclass, fields
from pathlib import Path

_KEY_VALUE = re.compile(r"^([A-Za-z][A-Za-z -]*?):\s*(.*)$")
_RANGE = re.compile(r"(-?[\d.]+)°C min / (-?[\d.]+)°C max")
_AMBIENT = re.compile(r"(-?[\d.]+)°C ambient")
_QUANTITY = re.compile(r"^(\d+)\s+(\w+)$")
_ROUTE = re.compile(r"^(.+?) to (.+)$")
_DELAY = re.compile(r"^(\d+) days?$")
_STATUS = re.compile(r"^([A-Z]+)(?: \((.+)\))?$")


@dataclass(frozen=True)
class Shipment:
    shipment_id: str
    product: str
    batch: str
    quantity: int
    unit: str
    origin: str
    destination: str
    dates: str
    month: str
    status: str
    status_note: str
    temp_min: float
    temp_max: float
    temp_ambient: int
    breach: int
    breach_detail: str
    delay_days: int
    qa_reassessment: str
    accepted_by: str
    action: str


def parse_shipment_log(text: str) -> list[Shipment]:
    """One Shipment per record. Raises ValueError on a record that doesn't parse: a table that
    silently lost a row would give confidently wrong counts."""
    parts = re.split(r"(?m)^=== (.+?) ===\s*$", text)
    return [_parse_record(header, body) for header, body in zip(parts[1::2], parts[2::2],
                                                                strict=True)
            if header.startswith("SHIPMENT SH-")]


def _parse_record(header: str, body: str) -> Shipment:
    values, flags = {}, []
    for part in re.split(r"[\n|]", body):
        part = part.strip()
        if not part:
            continue
        match = _KEY_VALUE.match(part)
        if match:
            values[match[1].lower()] = match[2].strip()
        else:
            flags.append(part)
    try:
        quantity = _QUANTITY.match(values["quantity"])
        route = _ROUTE.match(values["route"])
        status = _STATUS.match(values["status"])
        temp = values["temperature"]
        temp_range, ambient = _RANGE.search(temp), _AMBIENT.search(temp)
        low, high = temp_range.groups() if temp_range else (ambient[1], ambient[1])
        delay = _DELAY.match(values.get("delay", "0 days"))
        notes = [f for f in flags if f != "No breach"]
        return Shipment(
            shipment_id=values["shipment id"], product=values["product"], batch=values["batch"],
            quantity=int(quantity[1]), unit=quantity[2],
            origin=route[1], destination=route[2],
            dates=values["dates"], month=values["dates"][:3].lower(),
            status=status[1], status_note=status[2] or "",
            temp_min=float(low), temp_max=float(high), temp_ambient=int(not temp_range),
            breach=int("breach" in values), breach_detail=values.get("breach", ""),
            delay_days=int(delay[1]), qa_reassessment=values.get("qa reassessment", ""),
            accepted_by=values.get("accepted by", ""),
            action=" | ".join(filter(None, [values.get("action", ""), *notes])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"could not parse shipment record '{header}': {exc!r}") from exc


# The only filters that exist. A query cannot reach a column that isn't named here.
_FILTERS = {
    "status": "status = ?",
    "product": "product = ?",
    "month": "month = ?",
    "breach": "breach = ?",
    "destination": "destination LIKE ? ESCAPE '\\'",
    "accepted_by": "accepted_by LIKE ? ESCAPE '\\'",
}
_LIKE_FILTERS = {"destination", "accepted_by"}
_READ_ONLY_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_ORDER = " ORDER BY shipment_id"


class ShipmentStore:
    def __init__(self, shipments: list[Shipment]):
        columns = [f.name for f in fields(Shipment)]
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"CREATE TABLE shipments ({', '.join(columns)})")
        self._conn.executemany(
            f"INSERT INTO shipments VALUES ({', '.join('?' * len(columns))})",
            [astuple(s) for s in shipments],
        )
        self._conn.commit()
        self._conn.execute("PRAGMA query_only = ON")
        self._conn.set_authorizer(self._authorize)
        self._lock = threading.Lock()  # API handlers run in a thread pool

    @classmethod
    def from_file(cls, path: Path) -> "ShipmentStore":
        return cls(parse_shipment_log(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def _authorize(action, arg1, arg2, dbname, source) -> int:
        allowed = action in _READ_ONLY_ACTIONS
        return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY

    def _select(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params)]

    @staticmethod
    def _where(filters: dict) -> tuple[str, tuple]:
        unknown = set(filters) - set(_FILTERS)
        if unknown:
            raise ValueError(f"unknown filter: {sorted(unknown)}")
        if not filters:
            return "", ()
        clauses, params = [], []
        for name, value in filters.items():
            clauses.append(_FILTERS[name])
            if name in _LIKE_FILTERS:
                escaped = str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                value = f"%{escaped}%"
            params.append(value)
        return " WHERE " + " AND ".join(clauses), tuple(params)

    def rows(self, filters: dict | None = None) -> list[dict]:
        where, params = self._where(filters or {})
        return self._select(f"SELECT * FROM shipments{where}{_ORDER}", params)

    def get(self, shipment_id: str) -> dict | None:
        found = self._select("SELECT * FROM shipments WHERE shipment_id = ?", (shipment_id,))
        return found[0] if found else None

    def count(self, filters: dict | None = None) -> int:
        where, params = self._where(filters or {})
        return self._select(f"SELECT COUNT(*) AS n FROM shipments{where}", params)[0]["n"]

    def total_quantity(self, filters: dict | None = None) -> list[dict]:
        """Grouped by unit: vials and pouches can't be added together."""
        where, params = self._where(filters or {})
        return self._select(
            f"SELECT unit, SUM(quantity) AS total, COUNT(*) AS shipments FROM shipments{where} "
            "GROUP BY unit ORDER BY unit", params)

    def by_status(self) -> list[dict]:
        return self._select(
            "SELECT status, COUNT(*) AS shipments FROM shipments GROUP BY status ORDER BY status")

    def hottest(self) -> list[dict]:
        return self._select(
            "SELECT * FROM shipments WHERE temp_max = (SELECT MAX(temp_max) FROM shipments)"
            + _ORDER)

    def coldest(self) -> list[dict]:
        return self._select(
            "SELECT * FROM shipments WHERE temp_min = (SELECT MIN(temp_min) FROM shipments)"
            + _ORDER)
