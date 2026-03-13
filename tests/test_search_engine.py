from app.search_engine import SearchEngine
import unittest

class TestSearchEngine(unittest.TestCase):

  def setUp(self):
    self.search_engine = SearchEngine()

  def test_list_documents(self):
    # No documents should be in the vector store
    self.assertEqual(self.search_engine.list_documents(), [])

  def test_search_empty_query(self):
    # Empty query should raise a ValueError
    with self.assertRaises(ValueError):
      self.search_engine.search("")

  def test_search_with_no_documents(self):
    # Search should return empty list if no documents are in the vector store
    self.assertEqual(self.search_engine.search("test"), [])

  def test_add_document(self):
    # Add a document to the vector store
    self.search_engine.add_document("mums_ab_organizational_structure.pdf")

  def test_list_documents_after_adding(self):
    # List documents should now contain the added document
    self.search_engine.add_document("mums_ab_organizational_structure.pdf")
    self.assertEqual(self.search_engine.list_documents(), ["mums_ab_organizational_structure.pdf"])

  def test_search_no_results(self):
    self.search_engine.add_document("mums_ab_organizational_structure.pdf")
    results = self.search_engine.search("thisqueryshouldnotmatchanything123")
    self.assertIsInstance(results, list)
    self.assertEqual(len(results), 0, "Should return empty list for gibberish")

  def test_search_returns_results(self):
    self.search_engine.add_document("mums_ab_organizational_structure.pdf")
    results = self.search_engine.search("Who is the CEO of Mums AB?")
    self.assertIsInstance(results, list)
    self.assertGreater(len(results), 0, "Search should return at least one result")