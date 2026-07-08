# from pymongo import MongoClient

# client = MongoClient("mongodb://localhost:27017/")
# db = client["symptom_checker"]

# users_collection = db["users"]

# history_collection = db["history"]





# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any, Dict, Iterable, List, Optional

# try:
#     from pymongo import MongoClient
# except Exception:
#     MongoClient = None


# ARTIFACTS_DIR = Path("artifacts")
# ARTIFACTS_DIR.mkdir(exist_ok=True)

# LOCAL_USERS_FILE = ARTIFACTS_DIR / "local_users.json"
# LOCAL_HISTORY_FILE = ARTIFACTS_DIR / "local_history.json"


# def _ensure_file(path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     if not path.exists():
#         path.write_text("[]", encoding="utf-8")


# def _read_json(path: Path) -> List[Dict[str, Any]]:
#     _ensure_file(path)
#     try:
#         data = json.loads(path.read_text(encoding="utf-8"))
#         return data if isinstance(data, list) else []
#     except Exception:
#         return []


# def _write_json(path: Path, data: List[Dict[str, Any]]) -> None:
#     _ensure_file(path)
#     path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# def _match(document: Dict[str, Any], query: Optional[Dict[str, Any]]) -> bool:
#     if not query:
#         return True
#     for key, value in query.items():
#         if document.get(key) != value:
#             return False
#     return True


# def _apply_projection(document: Dict[str, Any], projection: Optional[Dict[str, int]]) -> Dict[str, Any]:
#     if not projection:
#         return dict(document)

#     doc = dict(document)
#     include_keys = [k for k, v in projection.items() if v]
#     exclude_keys = [k for k, v in projection.items() if not v]

#     if include_keys:
#         out = {k: doc.get(k) for k in include_keys if k in doc}
#         if projection.get("_id", 1) and "_id" in doc:
#             out["_id"] = doc["_id"]
#         return out

#     for key in exclude_keys:
#         doc.pop(key, None)
#     return doc


# class LocalCursor:
#     def __init__(self, rows: Iterable[Dict[str, Any]]):
#         self._rows = list(rows)

#     def sort(self, field: str, direction: int = 1):
#         reverse = direction == -1
#         self._rows.sort(key=lambda row: str(row.get(field, "")), reverse=reverse)
#         return self

#     def __iter__(self):
#         return iter(self._rows)

#     def __len__(self):
#         return len(self._rows)


# class LocalCollection:
#     def __init__(self, path: Path):
#         self.path = path
#         _ensure_file(self.path)

#     def find_one(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
#         for row in _read_json(self.path):
#             if _match(row, query):
#                 return _apply_projection(row, projection)
#         return None

#     def find(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
#         rows = []
#         for row in _read_json(self.path):
#             if _match(row, query):
#                 rows.append(_apply_projection(row, projection))
#         return LocalCursor(rows)

#     def insert_one(self, document: Dict[str, Any]):
#         rows = _read_json(self.path)
#         rows.append(dict(document))
#         _write_json(self.path, rows)
#         return {"acknowledged": True}


# def _build_mongo_collection(db_name: str, collection_name: str):
#     if MongoClient is None:
#         return None
#     try:
#         client = MongoClient(
#             "mongodb://localhost:27017/",
#             serverSelectionTimeoutMS=1200,
#             connectTimeoutMS=1200,
#             socketTimeoutMS=1200,
#         )
#         client.admin.command("ping")
#         db = client[db_name]
#         print("✓ MongoDB connected")
#         return db[collection_name]
#     except Exception as exc:
#         print(f"⚠ MongoDB unavailable, using local storage: {exc}")
#         return None


# class HybridCollection:
#     def __init__(self, mongo_collection, local_collection: LocalCollection):
#         self.mongo = mongo_collection
#         self.local = local_collection

#     def find_one(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
#         if self.mongo is not None:
#             try:
#                 return self.mongo.find_one(query or {}, projection)
#             except Exception as exc:
#                 print(f"Mongo find_one fallback: {exc}")
#                 self.mongo = None
#         return self.local.find_one(query, projection)

#     def find(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
#         if self.mongo is not None:
#             try:
#                 return self.mongo.find(query or {}, projection)
#             except Exception as exc:
#                 print(f"Mongo find fallback: {exc}")
#                 self.mongo = None
#         return self.local.find(query, projection)

#     def insert_one(self, document: Dict[str, Any]):
#         if self.mongo is not None:
#             try:
#                 return self.mongo.insert_one(document)
#             except Exception as exc:
#                 print(f"Mongo insert fallback: {exc}")
#                 self.mongo = None
#         return self.local.insert_one(document)


# users_collection = HybridCollection(
#     _build_mongo_collection("symptom_checker", "users"),
#     LocalCollection(LOCAL_USERS_FILE),
# )

# history_collection = HybridCollection(
#     _build_mongo_collection("symptom_checker", "history"),
#     LocalCollection(LOCAL_HISTORY_FILE),
# )









# from __future__ import annotations

# import json
# from datetime import date, datetime
# from pathlib import Path
# from typing import Any, Dict, Iterable, List, Optional

# try:
#     from pymongo import MongoClient
# except Exception:
#     MongoClient = None

# ARTIFACTS_DIR = Path("artifacts")
# ARTIFACTS_DIR.mkdir(exist_ok=True)
# LOCAL_USERS_FILE = ARTIFACTS_DIR / "local_users.json"
# LOCAL_HISTORY_FILE = ARTIFACTS_DIR / "local_history.json"

# def _ensure_file(path: Path) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     if not path.exists():
#         path.write_text("[]", encoding="utf-8")

# def _json_safe(value: Any) -> Any:
#     if isinstance(value, (datetime, date)):
#         return value.isoformat()
#     if isinstance(value, dict):
#         return {k: _json_safe(v) for k, v in value.items()}
#     if isinstance(value, list):
#         return [_json_safe(v) for v in value]
#     return value

# def _read_json(path: Path) -> List[Dict[str, Any]]:
#     _ensure_file(path)
#     try:
#         data = json.loads(path.read_text(encoding="utf-8"))
#         return data if isinstance(data, list) else []
#     except Exception:
#         return []

# def _write_json(path: Path, data: List[Dict[str, Any]]) -> None:
#     _ensure_file(path)
#     path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")

# def _match(document: Dict[str, Any], query: Optional[Dict[str, Any]]) -> bool:
#     if not query:
#         return True
#     for key, value in query.items():
#         if document.get(key) != value:
#             return False
#     return True

# def _apply_projection(document: Dict[str, Any], projection: Optional[Dict[str, int]]) -> Dict[str, Any]:
#     if not projection:
#         return dict(document)
#     doc = dict(document)
#     include_keys = [k for k, v in projection.items() if v]
#     exclude_keys = [k for k, v in projection.items() if not v]
#     if include_keys:
#         out = {k: doc.get(k) for k in include_keys if k in doc}
#         if projection.get("_id", 1) and "_id" in doc:
#             out["_id"] = doc["_id"]
#         return out
#     for key in exclude_keys:
#         doc.pop(key, None)
#     return doc

# class LocalCursor:
#     def __init__(self, rows: Iterable[Dict[str, Any]]):
#         self._rows = list(rows)
#     def sort(self, field: str, direction: int = 1):
#         reverse = direction == -1
#         self._rows.sort(key=lambda row: str(row.get(field, "")), reverse=reverse)
#         return self
#     def __iter__(self):
#         return iter(self._rows)

# class LocalCollection:
#     def __init__(self, path: Path):
#         self.path = path
#         _ensure_file(self.path)
#     def find_one(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
#         for row in _read_json(self.path):
#             if _match(row, query):
#                 return _apply_projection(row, projection)
#         return None
#     def find(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
#         rows = []
#         for row in _read_json(self.path):
#             if _match(row, query):
#                 rows.append(_apply_projection(row, projection))
#         return LocalCursor(rows)
#     def insert_one(self, document: Dict[str, Any]):
#         rows = _read_json(self.path)
#         rows.append(_json_safe(dict(document)))
#         _write_json(self.path, rows)
#         return {"acknowledged": True}

# def _build_mongo_collection(db_name: str, collection_name: str):
#     if MongoClient is None:
#         return None
#     try:
#         client = MongoClient(
#             "mongodb://localhost:27017/",
#             serverSelectionTimeoutMS=1200,
#             connectTimeoutMS=1200,
#             socketTimeoutMS=1200,
#         )
#         client.admin.command("ping")
#         db = client[db_name]
#         print("✓ MongoDB connected")
#         return db[collection_name]
#     except Exception as exc:
#         print(f"⚠ MongoDB unavailable, using local storage: {exc}")
#         return None

# class HybridCollection:
#     def __init__(self, mongo_collection, local_collection: LocalCollection):
#         self.mongo = mongo_collection
#         self.local = local_collection
#     def find_one(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
#         if self.mongo is not None:
#             try:
#                 return self.mongo.find_one(query or {}, projection)
#             except Exception as exc:
#                 print(f"Mongo find_one fallback: {exc}")
#                 self.mongo = None
#         return self.local.find_one(query, projection)
#     def find(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
#         if self.mongo is not None:
#             try:
#                 return self.mongo.find(query or {}, projection)
#             except Exception as exc:
#                 print(f"Mongo find fallback: {exc}")
#                 self.mongo = None
#         return self.local.find(query, projection)
#     def insert_one(self, document: Dict[str, Any]):
#         if self.mongo is not None:
#             try:
#                 return self.mongo.insert_one(document)
#             except Exception as exc:
#                 print(f"Mongo insert fallback: {exc}")
#                 self.mongo = None
#         return self.local.insert_one(document)

# users_collection = HybridCollection(_build_mongo_collection("symptom_checker", "users"), LocalCollection(LOCAL_USERS_FILE))
# history_collection = HybridCollection(_build_mongo_collection("symptom_checker", "history"), LocalCollection(LOCAL_HISTORY_FILE))





from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None


# IMPORTANT:
# Store local fallback data OUTSIDE the project folder so VS Code Live Server
# does not auto-refresh the page whenever history is saved.
APP_DIR = Path(tempfile.gettempdir()) / "symptosure_local_store"
APP_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_USERS_FILE = APP_DIR / "local_users.json"
LOCAL_HISTORY_FILE = APP_DIR / "local_history.json"


def _ensure_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _read_json(path: Path) -> List[Dict[str, Any]]:
    _ensure_file(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    _ensure_file(path)
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _match(document: Dict[str, Any], query: Optional[Dict[str, Any]]) -> bool:
    if not query:
        return True
    for key, value in query.items():
        if document.get(key) != value:
            return False
    return True


def _apply_projection(document: Dict[str, Any], projection: Optional[Dict[str, int]]) -> Dict[str, Any]:
    if not projection:
        return dict(document)

    doc = dict(document)
    include_keys = [k for k, v in projection.items() if v]
    exclude_keys = [k for k, v in projection.items() if not v]

    if include_keys:
        out = {k: doc.get(k) for k in include_keys if k in doc}
        if projection.get("_id", 1) and "_id" in doc:
            out["_id"] = doc["_id"]
        return out

    for key in exclude_keys:
        doc.pop(key, None)
    return doc


class LocalCursor:
    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self._rows = list(rows)

    def sort(self, field: str, direction: int = 1):
        reverse = direction == -1
        self._rows.sort(key=lambda row: str(row.get(field, "")), reverse=reverse)
        return self

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


class LocalCollection:
    def __init__(self, path: Path):
        self.path = path
        _ensure_file(self.path)

    def find_one(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
        for row in _read_json(self.path):
            if _match(row, query):
                return _apply_projection(row, projection)
        return None

    def find(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
        rows = []
        for row in _read_json(self.path):
            if _match(row, query):
                rows.append(_apply_projection(row, projection))
        return LocalCursor(rows)

    def insert_one(self, document: Dict[str, Any]):
        rows = _read_json(self.path)
        rows.append(_json_safe(dict(document)))
        _write_json(self.path, rows)
        return {"acknowledged": True}


def _build_mongo_collection(db_name: str, collection_name: str):
    if MongoClient is None:
        return None
    try:
        client = MongoClient(
            "mongodb://localhost:27017/",
            serverSelectionTimeoutMS=1200,
            connectTimeoutMS=1200,
            socketTimeoutMS=1200,
        )
        client.admin.command("ping")
        db = client[db_name]
        print("✓ MongoDB connected")
        return db[collection_name]
    except Exception as exc:
        print(f"⚠ MongoDB unavailable, using local storage: {exc}")
        print(f"✓ Local fallback path: {APP_DIR}")
        return None


class HybridCollection:
    def __init__(self, mongo_collection, local_collection: LocalCollection):
        self.mongo = mongo_collection
        self.local = local_collection

    def find_one(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
        if self.mongo is not None:
            try:
                return self.mongo.find_one(query or {}, projection)
            except Exception as exc:
                print(f"Mongo find_one fallback: {exc}")
                self.mongo = None
        return self.local.find_one(query, projection)

    def find(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, int]] = None):
        if self.mongo is not None:
            try:
                return self.mongo.find(query or {}, projection)
            except Exception as exc:
                print(f"Mongo find fallback: {exc}")
                self.mongo = None
        return self.local.find(query, projection)

    def insert_one(self, document: Dict[str, Any]):
        if self.mongo is not None:
            try:
                return self.mongo.insert_one(document)
            except Exception as exc:
                print(f"Mongo insert fallback: {exc}")
                self.mongo = None
        return self.local.insert_one(document)


users_collection = HybridCollection(
    _build_mongo_collection("symptom_checker", "users"),
    LocalCollection(LOCAL_USERS_FILE),
)

history_collection = HybridCollection(
    _build_mongo_collection("symptom_checker", "history"),
    LocalCollection(LOCAL_HISTORY_FILE),
)
