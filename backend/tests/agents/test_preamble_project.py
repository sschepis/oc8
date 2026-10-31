"""The project workspace must be unambiguous to every agent.

WHY THIS EXISTS. The operator works through one project folder checked into
GitHub, and should not have to re-explain "the project" to every agent or
pre-assign the file tool each request implies. This block is injected into
every run's preamble and names the repository, branch, local checkout, and the
request-as-authorization rule. These tests pin the contract.
"""

from __future__ import annotations

from types import SimpleNamespace

from oc8.agent.preamble import project_context_block


def test_project_block_names_repository_and_request_authorization() -> None:
    settings = SimpleNamespace(
        project_name="Theta Instruments",
        project_repo="sschepis/thetainstruments",
        project_branch="master",
        project_root="/Users/sschepis/Development/thetainstruments",
        project_description="NX-32/Nimbus hardware and company OS.",
    )
    text = project_context_block(settings)
    assert text is not None
    assert "Theta Instruments" in text
    assert "sschepis/thetainstruments" in text
    assert "master" in text
    assert 'When the user says "the project"' in text
    assert "The user's request is the authorization" in text
    assert "Do not ask the user to assign a tool" in text
    assert "/Users/sschepis/Development/thetainstruments" in text


def test_unscoped_deployment_has_no_project_block() -> None:
    settings = SimpleNamespace(
        project_name="",
        project_repo="",
        project_branch="main",
        project_root="",
        project_description="",
    )
    assert project_context_block(settings) is None
