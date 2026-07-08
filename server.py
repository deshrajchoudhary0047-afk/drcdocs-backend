from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import io
import json
import base64
import logging
import uuid
import re
import google.generativeai as genai
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

app = FastAPI(title="DRCDocs Enterprise API")
api = APIRouter(prefix="/api")


# =========================
# Models
# =========================
class OCRRequest(BaseModel):
    image_base64: str
    mime_type: str = "image/jpeg"


class OCRField(BaseModel):
    value: str = ""
    confidence: float = 0.0


class OCRResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_type: str = "unknown"
    raw_text: str = ""
    fields: Dict[str, OCRField] = {}
    overall_confidence: float = 0.0


class RecordCreate(BaseModel):
    # === CORE 10 FIELDS (per spec) ===
    date: str = ""
    lr_number: str = ""
    vehicle_number: str = ""
    supplier_gstin: str = ""
    supplier_name: str = ""
    dispatch_place: str = ""
    recipient_gstin: str = ""
    recipient_name: str = ""
    delivery_place: str = ""
    amount: str = ""
    # === Legacy / retained (populated only if present) ===
    document_type: str = "lr_receipt"
    document_number: str = ""  # legacy; mirror of lr_number
    gstin: str = ""            # legacy
    transporter_name: str = ""
    consignor: str = ""
    consignee: str = ""
    weight: str = ""
    distance: str = ""
    origin_address: str = ""
    destination_address: str = ""
    origin_city: str = ""
    destination_city: str = ""
    origin_state: str = ""
    destination_state: str = ""
    origin_pin: str = ""
    destination_pin: str = ""
    source: str = ""
    destination: str = ""
    remarks: str = ""
    verification_status: str = "pending"
    raw_text: str = ""
    image_base64: str = ""
    original_image_base64: str = ""
    processed_image_base64: str = ""
    pdf_base64: str = ""
    image_hash: str = ""
    file_paths: Dict[str, str] = {}
    folder_id: Optional[str] = None
    confidence_scores: Dict[str, float] = {}
    is_favorite: bool = False


class Record(RecordCreate):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    serial_no: int = 0  # auto-incremented on insert
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RecordUpdate(BaseModel):
    # Core
    date: Optional[str] = None
    lr_number: Optional[str] = None
    vehicle_number: Optional[str] = None
    supplier_gstin: Optional[str] = None
    supplier_name: Optional[str] = None
    dispatch_place: Optional[str] = None
    recipient_gstin: Optional[str] = None
    recipient_name: Optional[str] = None
    delivery_place: Optional[str] = None
    amount: Optional[str] = None
    # Legacy / retained
    document_type: Optional[str] = None
    document_number: Optional[str] = None
    gstin: Optional[str] = None
    transporter_name: Optional[str] = None
    consignor: Optional[str] = None
    consignee: Optional[str] = None
    weight: Optional[str] = None
    distance: Optional[str] = None
    origin_address: Optional[str] = None
    destination_address: Optional[str] = None
    origin_city: Optional[str] = None
    destination_city: Optional[str] = None
    origin_state: Optional[str] = None
    destination_state: Optional[str] = None
    origin_pin: Optional[str] = None
    destination_pin: Optional[str] = None
    source: Optional[str] = None
    destination: Optional[str] = None
    remarks: Optional[str] = None
    verification_status: Optional[str] = None
    folder_id: Optional[str] = None
    is_favorite: Optional[bool] = None
    raw_text: Optional[str] = None
    image_base64: Optional[str] = None
    processed_image_base64: Optional[str] = None
    pdf_base64: Optional[str] = None
    image_hash: Optional[str] = None
    file_paths: Optional[Dict[str, str]] = None
    confidence_scores: Optional[Dict[str, float]] = None


class FolderCreate(BaseModel):
    name: str
    color: str = "#FFB300"


class Folder(FolderCreate):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    record_count: int = 0


class ExportRequest(BaseModel):
    record_ids: Optional[List[str]] = None
    format: str = "csv"  # csv, json, txt, xlsx, pdf
    # Scope filters (all optional). If none provided and record_ids empty, exports ALL.
    folder_id: Optional[str] = None
    scope: Optional[str] = None  # today | week | month | year | all
    year: Optional[int] = None
    month: Optional[int] = None   # 1-12
    week: Optional[int] = None    # ISO week number (used with year)


class DuplicateRequest(BaseModel):
    lr_number: Optional[str] = ""
    vehicle_number: Optional[str] = ""
    supplier_gstin: Optional[str] = ""
    recipient_gstin: Optional[str] = ""
    amount: Optional[str] = ""
    date: Optional[str] = ""
    image_hash: Optional[str] = ""
    exclude_id: Optional[str] = None  # ignore this record (used when editing)


# =========================
# OCR via Emergent LLM
# =========================
SYSTEM_PROMPT = """You are an expert OCR + extraction engine for Indian transport & logistics documents (LR/bilty, e-way bills, GST tax invoices, delivery challans).

Return ONLY valid JSON. No prose, no markdown fences.
Schema:
{
 "document_type": "lr_receipt|bilty|eway_bill|gst_invoice|delivery_challan|invoice|unknown",
 "raw_text": "full raw OCR text",
 "fields": {
   "date": {"value": "", "confidence": 0.0},
   "lr_number": {"value": "", "confidence": 0.0},
   "vehicle_number": {"value": "", "confidence": 0.0},
   "supplier_gstin": {"value": "", "confidence": 0.0},
   "supplier_name": {"value": "", "confidence": 0.0},
   "dispatch_place": {"value": "", "confidence": 0.0},
   "recipient_gstin": {"value": "", "confidence": 0.0},
   "recipient_name": {"value": "", "confidence": 0.0},
   "delivery_place": {"value": "", "confidence": 0.0},
   "amount": {"value": "", "confidence": 0.0}
 },
 "overall_confidence": 0.0
}

Field mapping guide:
- lr_number: Lorry Receipt No / LR No / Consignment Note No / GC No / Bilty No / Invoice No (whichever is the primary document reference).
- supplier / consignor / sender / from party -> supplier_name & supplier_gstin.
- recipient / consignee / buyer / to party -> recipient_name & recipient_gstin.
- dispatch_place: origin city / from / dispatched from (city name).
- delivery_place: destination city / to / delivery at (city name).
- date: document date in the format printed on the document (dd/mm/yyyy or dd-mm-yyyy).
- amount: total amount / grand total as a plain number string (no currency symbol, no commas).
- Confidence 0.0-1.0. If a field is absent leave value:"" and confidence:0.0. Never invent values.
- GSTIN is a 15-character alphanumeric code (e.g., 27AAAAA0000A1Z5). Return only if clearly visible.
"""


def _extract_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

async def run_ocr(image_b64: str, mime_type: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Gemini API key not configured")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = SYSTEM_PROMPT

    try:
        image_bytes = base64.b64decode(image_b64)

        response = model.generate_content(
    [
        prompt,
        {
            "mime_type": mime_type,
            "data": image_bytes,
        },
    ],
    generation_config={
        "temperature": 0
    }
)
        data = _extract_json(response.text or "{}")

        default_fields = [
            "date", "lr_number", "vehicle_number",
            "supplier_gstin", "supplier_name", "dispatch_place",
            "recipient_gstin", "recipient_name", "delivery_place",
            "amount",
        ]

        fields = data.get("fields", {}) or {}

        for f in default_fields:
            if f not in fields or not isinstance(fields[f], dict):
                fields[f] = {"value": "", "confidence": 0.0}

            fields[f]["value"] = str(fields[f].get("value", "") or "")
            fields[f]["confidence"] = float(fields[f].get("confidence", 0.0) or 0.0)

        return {
            "document_type": data.get("document_type", "lr_receipt"),
            "raw_text": data.get("raw_text", ""),
            "fields": fields,
            "overall_confidence": float(data.get("overall_confidence", 0.0) or 0.0)
        }

    except Exception as e:
        logging.exception("Gemini OCR failed")
        raise HTTPException(500, f"OCR failed: {e}")


# =========================
# Routes
# =========================
@api.get("/")
async def root():
    return {"message": "DRCDocs Enterprise API", "version": "1.0.0"}


@api.get("/stats")
async def stats():
    total = await db.records.count_documents({})
    favorites = await db.records.count_documents({"is_favorite": True})
    folders = await db.folders.count_documents({})
    recent = await db.records.find({}, {"_id": 0}).sort("created_at", -1).limit(5).to_list(5)
    # count by type
    pipeline = [{"$group": {"_id": "$document_type", "count": {"$sum": 1}}}]
    by_type_cursor = db.records.aggregate(pipeline)
    by_type = {}
    async for row in by_type_cursor:
        by_type[row["_id"] or "unknown"] = row["count"]
    return {
        "total_records": total,
        "favorites": favorites,
        "folders": folders,
        "by_type": by_type,
        "recent": recent,
    }


@api.post("/ocr/scan")
async def ocr_scan(req: OCRRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key not configured"
        )

    if not req.image_base64:
        raise HTTPException(
            status_code=400,
            detail="image_base64 required"
        )

    try:
        result = await run_ocr(
            req.image_base64,
            req.mime_type
        )

        return JSONResponse(
            status_code=200,
            content=result
        )

    except HTTPException:
        raise

    except Exception as e:
        logging.exception("OCR Scan Error")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@api.post("/records", response_model=Record)
async def create_record(payload: RecordCreate):
    rec = Record(**payload.dict())
    # Mirror lr_number into legacy document_number for older exports
    if rec.lr_number and not rec.document_number:
        rec.document_number = rec.lr_number
    # Auto Sr No — atomic counter (safe under concurrent inserts)
    counter = await db.counters.find_one_and_update(
        {"_id": "records_serial"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True,
    )
    # Ensure counter is at least (max existing serial_no + 1) to survive backfills
    if not counter or counter.get("seq", 0) <= 0:
        last = await db.records.find_one({}, {"_id": 0, "serial_no": 1}, sort=[("serial_no", -1)])
        base = int((last or {}).get("serial_no") or 0)
        counter = await db.counters.find_one_and_update(
            {"_id": "records_serial"},
            {"$set": {"seq": base + 1}},
            upsert=True,
            return_document=True,
        )
    rec.serial_no = int(counter["seq"])
    # Guard: if seq has drifted below current max (e.g. counter was freshly created), bump it
    existing_max = await db.records.find_one({}, {"_id": 0, "serial_no": 1}, sort=[("serial_no", -1)])
    exmax = int((existing_max or {}).get("serial_no") or 0)
    if rec.serial_no <= exmax:
        counter = await db.counters.find_one_and_update(
            {"_id": "records_serial"},
            {"$set": {"seq": exmax + 1}},
            return_document=True,
        )
        rec.serial_no = int(counter["seq"])
    doc = rec.dict()
    await db.records.insert_one(doc)
    doc.pop("_id", None)
    return rec


@api.get("/records")
async def list_records(
    q: Optional[str] = None,
    folder_id: Optional[str] = None,
    favorites: Optional[bool] = None,
    document_type: Optional[str] = None,
    scope: Optional[str] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    week: Optional[int] = None,
    limit: int = 200,
):
    query: Dict[str, Any] = {}
    if folder_id:
        query["folder_id"] = folder_id
    if favorites:
        query["is_favorite"] = True
    if document_type:
        query["document_type"] = document_type
    scope_q = _scope_query(scope, year, month, week)
    query.update(scope_q)
    if q:
        rx = {"$regex": re.escape(q), "$options": "i"}
        query["$or"] = [
            # Core 10
            {"lr_number": rx}, {"vehicle_number": rx},
            {"supplier_gstin": rx}, {"supplier_name": rx}, {"dispatch_place": rx},
            {"recipient_gstin": rx}, {"recipient_name": rx}, {"delivery_place": rx},
            {"amount": rx}, {"date": rx},
            # Legacy fallback
            {"document_number": rx}, {"gstin": rx}, {"transporter_name": rx},
            {"consignor": rx}, {"consignee": rx},
            {"source": rx}, {"destination": rx}, {"remarks": rx}, {"raw_text": rx},
        ]
    cursor = db.records.find(
        query,
        {"_id": 0, "image_base64": 0, "original_image_base64": 0, "processed_image_base64": 0, "pdf_base64": 0},
    ).sort("created_at", -1).limit(limit)
    items = await cursor.to_list(limit)
    return {"items": items, "count": len(items)}


@api.get("/records/{record_id}")
async def get_record(record_id: str):
    rec = await db.records.find_one({"id": record_id}, {"_id": 0})
    if not rec:
        raise HTTPException(404, "Record not found")
    return rec


@api.patch("/records/{record_id}")
async def update_record(record_id: str, patch: RecordUpdate):
    updates = {k: v for k, v in patch.dict().items() if v is not None}
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    res = await db.records.update_one({"id": record_id}, {"$set": updates})
    if res.matched_count == 0:
        raise HTTPException(404, "Record not found")
    rec = await db.records.find_one({"id": record_id}, {"_id": 0})
    return rec


@api.delete("/records/{record_id}")
async def delete_record(record_id: str):
    res = await db.records.delete_one({"id": record_id})
    if res.deleted_count == 0:
        raise HTTPException(404, "Record not found")
    return {"ok": True}


@api.post("/records/duplicates")
async def check_duplicates(payload: DuplicateRequest):
    ors: List[Dict[str, Any]] = []
    lr = (payload.lr_number or "").strip()
    if lr:
        # Match by lr_number or legacy document_number
        ors.append({"lr_number": lr})
        ors.append({"document_number": lr})
    for key in ("vehicle_number", "supplier_gstin", "recipient_gstin", "image_hash"):
        val = (getattr(payload, key) or "").strip()
        if val:
            ors.append({key: val})
    amt = (payload.amount or "").strip()
    dt = (payload.date or "").strip()
    if amt and dt:
        ors.append({"$and": [{"amount": amt}, {"date": dt}]})
    if not ors:
        return {"duplicates": []}
    q: Dict[str, Any] = {"$or": ors}
    if payload.exclude_id:
        q["id"] = {"$ne": payload.exclude_id}
    dups = await db.records.find(
        q,
        {"_id": 0, "image_base64": 0, "original_image_base64": 0, "processed_image_base64": 0, "pdf_base64": 0},
    ).limit(20).to_list(20)
    return {"duplicates": dups}


@api.post("/records/{record_id}/move")
async def move_record(record_id: str, payload: Dict[str, Any]):
    folder_id = payload.get("folder_id")  # can be None to unassign
    res = await db.records.update_one(
        {"id": record_id},
        {"$set": {"folder_id": folder_id, "updated_at": datetime.now(timezone.utc).isoformat()}},
    )
    if res.matched_count == 0:
        raise HTTPException(404, "Record not found")
    return {"ok": True, "folder_id": folder_id}


# ----- Folders -----
@api.post("/folders", response_model=Folder)
async def create_folder(payload: FolderCreate):
    folder = Folder(**payload.dict())
    await db.folders.insert_one(folder.dict())
    return folder


@api.get("/folders")
async def list_folders():
    folders = await db.folders.find({}, {"_id": 0}).sort("created_at", -1).to_list(200)
    for f in folders:
        f["record_count"] = await db.records.count_documents({"folder_id": f["id"]})
    return {"items": folders}


def _iso_now() -> datetime:
    return datetime.now(timezone.utc)


def _scope_query(scope: Optional[str], year: Optional[int], month: Optional[int], week: Optional[int]) -> Dict[str, Any]:
    """Build a Mongo query for a smart date scope on `created_at` (ISO string)."""
    now = _iso_now()
    if scope == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        return {"created_at": {"$gte": start}}
    if scope == "week":
        # Rolling 7 days
        from datetime import timedelta
        start = (now - timedelta(days=7)).isoformat()
        return {"created_at": {"$gte": start}}
    if scope == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        return {"created_at": {"$gte": start}}
    if scope == "year":
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        return {"created_at": {"$gte": start}}
    if year and week:
        # ISO week filtering via python date arithmetic
        from datetime import date, timedelta
        try:
            monday = date.fromisocalendar(year, week, 1)
            next_monday = monday + timedelta(days=7)
            return {"created_at": {"$gte": monday.isoformat(), "$lt": next_monday.isoformat()}}
        except Exception:
            return {}
    if year and month:
        y_start = f"{year:04d}-{month:02d}-01T00:00:00"
        nm = month + 1
        ny = year
        if nm > 12:
            nm = 1
            ny = year + 1
        y_end = f"{ny:04d}-{nm:02d}-01T00:00:00"
        return {"created_at": {"$gte": y_start, "$lt": y_end}}
    if year:
        return {"created_at": {"$gte": f"{year:04d}-01-01T00:00:00", "$lt": f"{year + 1:04d}-01-01T00:00:00"}}
    return {}


@api.get("/folders/smart")
async def smart_folders():
    """Auto-organized folders + year/month breakdown of all records."""
    all_count = await db.records.count_documents({})
    smart = []
    for key, label in [("today", "Today"), ("week", "This Week"), ("month", "This Month"), ("year", "This Year")]:
        q = _scope_query(key, None, None, None)
        smart.append({"key": key, "label": label, "count": await db.records.count_documents(q)})
    smart.append({"key": "all", "label": "All Records", "count": all_count})

    # Year -> Month breakdown from created_at ISO strings
    pipeline = [
        {"$match": {"created_at": {"$exists": True, "$ne": ""}}},
        {"$project": {
            "year": {"$substrBytes": ["$created_at", 0, 4]},
            "month": {"$substrBytes": ["$created_at", 5, 2]},
        }},
        {"$group": {"_id": {"year": "$year", "month": "$month"}, "count": {"$sum": 1}}},
        {"$sort": {"_id.year": -1, "_id.month": -1}},
    ]
    year_map: Dict[str, Dict[str, Any]] = {}
    async for row in db.records.aggregate(pipeline):
        y = row["_id"]["year"]
        m = row["_id"]["month"]
        year_map.setdefault(y, {"year": y, "total": 0, "months": []})
        year_map[y]["months"].append({"month": m, "count": row["count"]})
        year_map[y]["total"] += row["count"]

    return {"smart": smart, "years": list(year_map.values())}


@api.delete("/folders/{folder_id}")
async def delete_folder(folder_id: str):
    await db.folders.delete_one({"id": folder_id})
    await db.records.update_many({"folder_id": folder_id}, {"$set": {"folder_id": None}})
    return {"ok": True}


# ----- Export -----
CORE_COLUMNS = [
    ("serial_no", "Sr No."),
    ("date", "Date"),
    ("lr_number", "LR Number"),
    ("vehicle_number", "Vehicle Number"),
    ("supplier_gstin", "Supplier GSTIN"),
    ("supplier_name", "Supplier Name"),
    ("dispatch_place", "Dispatch Place"),
    ("recipient_gstin", "Recipient GSTIN"),
    ("recipient_name", "Recipient Name"),
    ("delivery_place", "Delivery Place"),
    ("amount", "Amount"),
]


def _core_row(r: Dict[str, Any], idx: int) -> List[str]:
    row = []
    for key, _ in CORE_COLUMNS:
        if key == "serial_no":
            row.append(str(r.get("serial_no") or idx))
        else:
            # Fallbacks for older records
            v = r.get(key) or ""
            if not v:
                if key == "lr_number":
                    v = r.get("document_number") or ""
                elif key == "supplier_gstin":
                    v = r.get("gstin") or ""
                elif key == "supplier_name":
                    v = r.get("consignor") or r.get("transporter_name") or ""
                elif key == "dispatch_place":
                    v = r.get("source") or r.get("origin_city") or ""
                elif key == "recipient_name":
                    v = r.get("consignee") or ""
                elif key == "delivery_place":
                    v = r.get("destination") or r.get("destination_city") or ""
            row.append(str(v))
    return row


def _build_csv(records: List[Dict[str, Any]]) -> bytes:
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([label for _, label in CORE_COLUMNS])
    for idx, r in enumerate(records, start=1):
        w.writerow(_core_row(r, idx))
    return buf.getvalue().encode("utf-8")


def _build_txt(records: List[Dict[str, Any]]) -> bytes:
    header = " | ".join(label for _, label in CORE_COLUMNS)
    lines = [header, "-" * len(header)]
    for idx, r in enumerate(records, start=1):
        lines.append(" | ".join(_core_row(r, idx)))
    return "\n".join(lines).encode("utf-8")


def _build_xlsx(records: List[Dict[str, Any]]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    wb = Workbook()
    ws = wb.active
    ws.title = "DRCDocs"
    labels = [label for _, label in CORE_COLUMNS]
    ws.append(labels)
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1A1D24", end_color="1A1D24", fill_type="solid")
    thin = Side(border_style="thin", color="CCCCCC")
    border = Border(top=thin, left=thin, right=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border
    for idx, r in enumerate(records, start=1):
        row = _core_row(r, idx)
        ws.append(row)
        for cell in ws[ws.max_row]:
            cell.border = border
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    widths = [8, 14, 18, 18, 22, 26, 22, 22, 26, 22, 14]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    ws.freeze_panes = "A2"
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def _build_pdf(records: List[Dict[str, Any]]) -> bytes:
    """Render records in the same tabular layout as the Excel sheet."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import mm
    bio = io.BytesIO()
    doc = SimpleDocTemplate(
        bio, pagesize=landscape(A4),
        leftMargin=10 * mm, rightMargin=10 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>DRCDocs Enterprise — Records Ledger</b>", styles["Title"]),
        Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}  |  Total: {len(records)}", styles["Normal"]),
        Spacer(1, 6),
    ]
    headers = [label for _, label in CORE_COLUMNS]
    data = [headers]
    for idx, r in enumerate(records, start=1):
        data.append(_core_row(r, idx))
    col_widths = [14, 22, 30, 30, 36, 42, 36, 36, 42, 36, 24]
    col_widths = [w * mm for w in col_widths]
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1A1D24")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.whitesmoke, rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    if not records:
        story.append(Paragraph("<i>No records to export.</i>", styles["Normal"]))
    doc.build(story)
    return bio.getvalue()


@api.post("/export")
async def export_records(req: ExportRequest):
    query: Dict[str, Any] = {}
    if req.record_ids:
        query["id"] = {"$in": req.record_ids}
    else:
        if req.folder_id:
            query["folder_id"] = req.folder_id
        scope_q = _scope_query(req.scope, req.year, req.month, req.week)
        query.update(scope_q)
    records = await db.records.find(
        query,
        {"_id": 0, "image_base64": 0, "original_image_base64": 0, "processed_image_base64": 0, "pdf_base64": 0},
    ).sort("serial_no", 1).to_list(5000)
    fmt = (req.format or "csv").lower()
    if fmt == "csv":
        data = _build_csv(records)
        mime = "text/csv"
        ext = "csv"
    elif fmt == "json":
        data = json.dumps(records, indent=2, default=str).encode("utf-8")
        mime = "application/json"
        ext = "json"
    elif fmt == "txt":
        data = _build_txt(records)
        mime = "text/plain"
        ext = "txt"
    elif fmt == "xlsx":
        data = _build_xlsx(records)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ext = "xlsx"
    elif fmt == "pdf":
        data = _build_pdf(records)
        mime = "application/pdf"
        ext = "pdf"
    else:
        raise HTTPException(400, "invalid format")
    b64 = base64.b64encode(data).decode("utf-8")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return {"filename": f"drcdocs-{ts}.{ext}", "mime_type": mime, "base64": b64, "count": len(records)}


app.include_router(api)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
