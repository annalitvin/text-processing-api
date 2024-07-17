from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.preprocessing.text_normalizer import TextNormalizer


@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c


TEXT = (
    "A traditional critical review includes an introduction summarizing the key details of the work being "
    "reviewed, the body containing the evaluation, and a conclusion summarizing the review. "
    "Instructional texts usually start with an overview of the task or goal, and possibly, "
    "what the end result should look like."
)


@pytest.fixture(params=[TEXT])
def text_normalizer(request):
    return TextNormalizer(request.param)
