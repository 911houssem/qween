import tempfile
import unittest
from pathlib import Path

from agent.rag.docs import DocsIndex


class DocsIndexTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.docs = DocsIndex(self.root / ".agent")

    def tearDown(self):
        self.td.cleanup()

    def test_changed_document_replaces_old_chunks(self):
        f = self.root / "guide.md"
        f.write_text("# Setup\\n\\nInstall the old package.", encoding="utf-8")
        self.docs.ingest(f)
        self.assertTrue(self.docs.search("old package"))
        f.write_text("# Setup\\n\\nInstall the new framework.", encoding="utf-8")
        self.docs.ingest(f)
        self.assertFalse(self.docs.search("old package"))
        self.assertTrue(self.docs.search("new framework"))

    def test_directory_ingest_prunes_deleted_documents(self):
        folder = self.root / "docs"
        folder.mkdir()
        f = folder / "guide.md"
        f.write_text("unique obsolete guidance", encoding="utf-8")
        self.docs.ingest(folder)
        self.assertTrue(self.docs.search("obsolete guidance"))
        f.unlink()
        self.docs.ingest(folder)
        self.assertFalse(self.docs.search("obsolete guidance"))


if __name__ == "__main__":
    unittest.main()
