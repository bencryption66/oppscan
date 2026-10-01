import pytest

from oppscan import db


@pytest.fixture
def con():
    connection = db.connect(":memory:")
    yield connection
    connection.close()
