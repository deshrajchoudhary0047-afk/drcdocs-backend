"""DRCDocs Enterprise API tests"""
import os
import base64
import io
import json
import pytest
import requests
from pathlib import Path

BASE_URL = "https://drcdocs-backend.onrender.com"

# Small real JPEG image (invoice-style with text) - we generate one with PIL
def _build_test_jpeg_b64():
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (800, 600), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
            small = font
        lines = [
            ("TAX INVOICE", font),
            ("Invoice No: INV-2026-001", small),
            ("Date: 05/01/2026", small),
            ("GSTIN: 27AAECS1234N1Z5", small),
            ("Transporter: SIDDHI LOGISTICS", small),
            ("Vehicle No: MH12AB1234", small),
            ("Consignor: ACME EXPORTS PVT LTD", small),
            ("Consignee: XYZ TRADERS", small),
            ("From: Mumbai, Maharashtra-400001", small),
            ("To: Delhi, Delhi-110001", small),
            ("Amount: 45250.00  Weight: 1200 kg", small),
        ]
        y = 30
        for text, f in lines:
            draw.text((30, y), text, fill="black", font=f)
            y += 40
        # add some visual features
        draw.rectangle([(20, 20), (780, 580)], outline="black", width=3)
        draw.line([(20, 80), (780, 80)], fill="black", width=2)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        pytest.skip(f"Cannot build test image: {e}")


@pytest.fixture(scope="session")
def api():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


@pytest.fixture(scope="session")
def created_ids():
    return {"records": [], "folders": []}


# -------- Health --------
class TestHealth:
    def test_root(self, api):
        r = api.get(f"{BASE_URL}/api/")
        assert r.status_code == 200
        assert "DRCDocs" in r.json().get("message", "")


# -------- Stats --------
class TestStats:
    def test_stats_shape(self, api):
        r = api.get(f"{BASE_URL}/api/stats")
        assert r.status_code == 200
        data = r.json()
        for k in ("total_records", "favorites", "folders", "by_type", "recent"):
            assert k in data
        assert isinstance(data["by_type"], dict)
        assert isinstance(data["recent"], list)
        # ensure no _id leaked
        for rec in data["recent"]:
            assert "_id" not in rec


# -------- Records CRUD --------
class TestRecordsCRUD:
    def test_create_record(self, api, created_ids):
        payload = {
            "document_type": "invoice",
            "document_number": "TEST_INV_001",
            "vehicle_number": "MH12AB9999",
            "gstin": "27TESTGST123",
            "transporter_name": "TEST Transporter",
            "consignor": "TEST Consignor",
            "consignee": "TEST Consignee",
            "amount": "1000.00",
        }
        r = api.post(f"{BASE_URL}/api/records", json=payload)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["document_number"] == "TEST_INV_001"
        assert "id" in d
        assert "_id" not in d
        created_ids["records"].append(d["id"])

    def test_get_record(self, api, created_ids):
        assert created_ids["records"], "no record created"
        rid = created_ids["records"][0]
        r = api.get(f"{BASE_URL}/api/records/{rid}")
        assert r.status_code == 200
        d = r.json()
        assert d["id"] == rid
        assert "_id" not in d

    def test_list_records(self, api):
        r = api.get(f"{BASE_URL}/api/records")
        assert r.status_code == 200
        d = r.json()
        assert "items" in d and "count" in d
        for it in d["items"]:
            assert "_id" not in it

    def test_search_query(self, api):
        r = api.get(f"{BASE_URL}/api/records", params={"q": "TEST_INV_001"})
        assert r.status_code == 200
        items = r.json()["items"]
        assert any(x["document_number"] == "TEST_INV_001" for x in items)

    def test_filter_document_type(self, api):
        r = api.get(f"{BASE_URL}/api/records", params={"document_type": "invoice"})
        assert r.status_code == 200
        for it in r.json()["items"]:
            assert it["document_type"] == "invoice"

    def test_patch_favorite_toggle(self, api, created_ids):
        rid = created_ids["records"][0]
        r = api.patch(f"{BASE_URL}/api/records/{rid}", json={"is_favorite": True})
        assert r.status_code == 200
        # verify via GET
        g = api.get(f"{BASE_URL}/api/records/{rid}")
        assert g.json()["is_favorite"] is True

    def test_filter_favorites(self, api, created_ids):
        r = api.get(f"{BASE_URL}/api/records", params={"favorites": "true"})
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()["items"]]
        assert created_ids["records"][0] in ids

    def test_duplicates(self, api):
        r = api.post(f"{BASE_URL}/api/records/duplicates",
                     json={"document_number": "TEST_INV_001"})
        assert r.status_code == 200
        dups = r.json()["duplicates"]
        assert len(dups) >= 1
        for d in dups:
            assert "_id" not in d

    def test_delete_record(self, api, created_ids):
        rid = created_ids["records"][0]
        r = api.delete(f"{BASE_URL}/api/records/{rid}")
        assert r.status_code == 200
        # verify 404 afterwards
        g = api.get(f"{BASE_URL}/api/records/{rid}")
        assert g.status_code == 404
        created_ids["records"].pop(0)


# -------- Folders --------
class TestFolders:
    def test_create_folder(self, api, created_ids):
        r = api.post(f"{BASE_URL}/api/folders",
                     json={"name": "TEST_Folder_A", "color": "#FF0000"})
        assert r.status_code == 200
        d = r.json()
        assert d["name"] == "TEST_Folder_A"
        assert "id" in d
        assert "_id" not in d
        created_ids["folders"].append(d["id"])

    def test_list_folders_has_record_count(self, api, created_ids):
        r = api.get(f"{BASE_URL}/api/folders")
        assert r.status_code == 200
        items = r.json()["items"]
        assert isinstance(items, list)
        found = [f for f in items if f["id"] in created_ids["folders"]]
        assert found
        for f in found:
            assert "record_count" in f
            assert "_id" not in f

    def test_delete_folder(self, api, created_ids):
        fid = created_ids["folders"][0]
        r = api.delete(f"{BASE_URL}/api/folders/{fid}")
        assert r.status_code == 200
        created_ids["folders"].pop(0)


# -------- Export --------
class TestExport:
    @pytest.mark.parametrize("fmt,mime_prefix", [
        ("csv", "text/csv"),
        ("json", "application/json"),
        ("txt", "text/plain"),
        ("xlsx", "application/vnd.openxml"),
        ("pdf", "application/pdf"),
    ])
    def test_export_format(self, api, fmt, mime_prefix):
        r = api.post(f"{BASE_URL}/api/export", json={"format": fmt})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["filename"].endswith(f".{fmt}")
        assert d["mime_type"].startswith(mime_prefix)
        assert "count" in d
        # decode base64 and check non-empty (unless zero records)
        raw = base64.b64decode(d["base64"])
        if d["count"] > 0 or fmt in ("json", "pdf"):
            assert len(raw) > 0
        # Format-specific validation
        if fmt == "pdf":
            assert raw[:4] == b"%PDF"
        elif fmt == "xlsx":
            assert raw[:2] == b"PK"  # zip signature
        elif fmt == "json":
            json.loads(raw.decode("utf-8"))


# -------- OCR --------
class TestOCR:
    def test_ocr_scan_returns_structured_json(self, api):
        b64 = _build_test_jpeg_b64()
        r = api.post(f"{BASE_URL}/api/ocr/scan",
                     json={"image_base64": b64, "mime_type": "image/jpeg"},
                     timeout=120)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "document_type" in d
        assert "raw_text" in d
        assert "fields" in d
        assert "overall_confidence" in d
        # Verify each required field has value/confidence
        for key in ("document_number", "vehicle_number", "gstin",
                    "transporter_name", "consignor", "consignee",
                    "date", "amount"):
            assert key in d["fields"], f"missing field {key}"
            assert "value" in d["fields"][key]
            assert "confidence" in d["fields"][key]
