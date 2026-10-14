# WriterAgent - AI Writing Assistant for LibreOffice
# Copyright (c) 2026 KeithCu
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native UNO integration tests for Writer track changes tools."""

from plugin.testing_runner import native_test
from plugin.tests.testing_utils import TestingFactory, with_native_doc


@native_test
@with_native_doc("writer")
def test_writer_track_changes_show_toggle_uno(ctx, doc):
    """TrackChangesShow sets RedlineDisplayType and updates ShowChanges without touching ShowChangesInMargin."""
    from plugin.main import get_services
    from plugin.writer.tracking import TrackChangesShow

    controller = doc.getCurrentController()
    view_settings = controller.getViewSettings()

    saved_display_type = doc.getPropertyValue("RedlineDisplayType")
    saved_margin = view_settings.getPropertyValue("ShowChangesInMargin")

    tctx = TestingFactory.create_context(
        doc=doc, ctx=ctx, env="native", doc_type="writer", services=get_services()
    )

    try:
        # show=False -> type 1, ShowChanges False
        res_hide = TrackChangesShow().execute(tctx, show=False)
        assert res_hide.get("status") == "ok", res_hide
        assert doc.getPropertyValue("RedlineDisplayType") == 1
        assert doc.getPropertyValue("ShowChanges") is False
        assert view_settings.getPropertyValue("ShowChangesInMargin") == saved_margin

        # show=True -> type 2, ShowChanges True
        res_show = TrackChangesShow().execute(tctx, show=True)
        assert res_show.get("status") == "ok", res_show
        assert doc.getPropertyValue("RedlineDisplayType") == 2
        assert doc.getPropertyValue("ShowChanges") is True
        assert view_settings.getPropertyValue("ShowChangesInMargin") == saved_margin
    finally:
        # Restore original values so reused document is left clean
        doc.setPropertyValue("RedlineDisplayType", saved_display_type)
        view_settings.setPropertyValue("ShowChangesInMargin", saved_margin)


@native_test
@with_native_doc("writer")
def test_accept_body_redline_with_redline_in_table_cell_uno(ctx, doc):
    """Cross-text redline compareRegionStarts IllegalArgumentException is not treated as overlap."""
    from plugin.main import get_services
    from plugin.writer.tracking import TrackChangesList, ManageTrackedChanges

    tctx = TestingFactory.create_context(
        doc=doc, ctx=ctx, env="native", doc_type="writer", services=get_services()
    )

    try:
        doc.setPropertyValue("RecordChanges", False)
        text = doc.getText()
        cursor = text.createTextCursor()
        cursor.gotoEnd(False)
        text.insertString(cursor, "hello ", False)

        table = doc.createInstance("com.sun.star.text.TextTable")
        table.initialize(1, 1)
        text.insertTextContent(cursor, table, False)
        cell = table.getCellByName("A1")
        cell.setString("BETA")

        doc.setPropertyValue("RecordChanges", True)
        cursor.gotoEnd(False)
        text.insertString(cursor, "ALPHA ", False)

        cell_cursor = cell.createTextCursor()
        cell_cursor.gotoEnd(False)
        cell.insertString(cell_cursor, "X", False)

        doc.setPropertyValue("RecordChanges", False)

        list_res = TrackChangesList().execute(tctx)
        assert list_res.get("status") == "ok", list_res
        changes = list_res.get("changes", [])
        alpha_index = None
        for ch in changes:
            if "ALPHA" in ch.get("text", ""):
                alpha_index = ch["index"]
                break
        assert alpha_index is not None, f"ALPHA change not found in {changes}"

        res = ManageTrackedChanges().execute(tctx, action="accept", index=alpha_index)
        assert res.get("status") == "ok", res

        list_res2 = TrackChangesList().execute(tctx)
        assert list_res2.get("status") == "ok", list_res2
        changes2 = list_res2.get("changes", [])
        assert not any("ALPHA" in ch.get("text", "") for ch in changes2)
        assert any("X" in ch.get("text", "") for ch in changes2)
    finally:
        doc.setPropertyValue("RecordChanges", False)
        try:
            ManageTrackedChanges().execute(tctx, action="reject_all")
        except Exception:
            pass

