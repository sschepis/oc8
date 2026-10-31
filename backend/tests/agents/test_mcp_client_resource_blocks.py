"""MCP content blocks must deliver file bodies, not just metadata.

WHY THIS EXISTS. GitHub's official MCP server returns `get_file_contents` as a
short `TextContent` summary ("successfully downloaded text file (SHA: ...)")
plus an `EmbeddedResource` whose `TextResourceContents` carries the actual file.
`McpSession.call()` used to read only `block.text`, so the model received the
SHA and never the file -- tools looked connected while being unusable for
reading source or documentation. These tests pin the block conversion for text,
embedded text, embedded text blobs, and resource links.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

from oc8.agent.mcp_client import _MAX_RESOURCE_CHARS, _content_block_text


def test_plain_text_content() -> None:
    assert _content_block_text(SimpleNamespace(text="hello")) == "hello"


def test_embedded_text_resource() -> None:
    block = SimpleNamespace(
        resource=SimpleNamespace(
            text="# Theta Instruments\n\nBody text.",
            uri="repo://example/README.md",
            mime_type="text/plain; charset=utf-8",
        )
    )
    assert _content_block_text(block) == "# Theta Instruments\n\nBody text."


def test_embedded_text_blob_is_decoded() -> None:
    blob = base64.b64encode(b"decoded body").decode()
    block = SimpleNamespace(
        resource=SimpleNamespace(
            blob=blob,
            uri="repo://example/file.txt",
            mime_type="text/plain",
        )
    )
    assert _content_block_text(block) == "decoded body"


def test_embedded_binary_resource_is_marked_not_dropped() -> None:
    block = SimpleNamespace(
        resource=SimpleNamespace(
            blob="AAEC",
            uri="repo://example/photo.png",
            mime_type="image/png",
        )
    )
    result = _content_block_text(block)
    assert result is not None
    assert "binary resource" in result
    assert "image/png" in result


def test_resource_link_is_preserved() -> None:
    block = SimpleNamespace(
        uri="repo://example/file.md",
        name="file.md",
        mime_type="text/markdown",
    )
    result = _content_block_text(block)
    assert result is not None
    assert "repo://example/file.md" in result
    assert "file.md" in result


def test_large_embedded_text_is_clipped_with_a_marker() -> None:
    body = "x" * (_MAX_RESOURCE_CHARS + 50)
    block = SimpleNamespace(resource=SimpleNamespace(text=body, uri="repo://big"))
    result = _content_block_text(block)
    assert result is not None
    assert len(result) < len(body)
    assert "truncated" in result


def test_unknown_block_is_ignored() -> None:
    assert _content_block_text(SimpleNamespace()) is None
