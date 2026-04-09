"""
API endpoint tests for GPU cluster
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from PIL import Image
import io

from api.main import app

client = TestClient(app)


def create_test_image(size=(512, 512), color=(255, 0, 0)):
    """Create a test image"""
    img = Image.new('RGB', size, color)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "triposr" in data["models"]
    assert "trellis2" in data["models"]


def test_get_categories():
    """Test categories endpoint"""
    response = client.get("/api/categories")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert len(data["categories"]) == 5

    # Check human_bust category
    human_bust = next(c for c in data["categories"] if c["category"] == "human_bust")
    assert human_bust["best_model"] == "both"
    assert human_bust["foreground_ratio"] == 0.80


def test_upload_job_triposr():
    """Test job upload with TripoSR"""
    test_img = create_test_image()

    response = client.post(
        "/api/upload",
        files={"file": ("test.png", test_img, "image/png")},
        data={
            "category": "human_bust",
            "model": "triposr"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued" or data["status"] == "preprocessing"


def test_upload_job_trellis2():
    """Test job upload with Trellis.2"""
    test_img = create_test_image()

    response = client.post(
        "/api/upload",
        files={"file": ("test.png", test_img, "image/png")},
        data={
            "category": "generic_object",
            "model": "trellis2"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data


def test_upload_invalid_category():
    """Test upload with invalid category"""
    test_img = create_test_image()

    response = client.post(
        "/api/upload",
        files={"file": ("test.png", test_img, "image/png")},
        data={
            "category": "invalid_category",
            "model": "triposr"
        }
    )

    assert response.status_code == 422  # Validation error


def test_upload_invalid_file():
    """Test upload with non-image file"""
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        data={
            "category": "human_bust",
            "model": "triposr"
        }
    )

    assert response.status_code == 400


def test_get_job_status_not_found():
    """Test getting status of non-existent job"""
    response = client.get("/api/job/nonexistent-job-id/status")
    assert response.status_code == 404


def test_get_preview_not_found():
    """Test getting preview of non-existent job"""
    response = client.get("/api/preview/nonexistent-job-id")
    assert response.status_code == 404


def test_queue_status():
    """Test queue status endpoint"""
    response = client.get("/api/queue/status")
    assert response.status_code == 200
    data = response.json()
    assert "triposr" in data
    assert "trellis2" in data
    assert "queue_length" in data["triposr"]
    assert "active_jobs" in data["triposr"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

