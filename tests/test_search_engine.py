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

  def test_semantic_search(self):
    # Embeddings should find related concepts, not just exact keyword matches
    self.search_engine.add_document("mums_ab_product_descriptions.pdf")
    results = self.search_engine.search("healthy snacks")
    self.assertGreater(len(results), 0, "Semantic search should find related products")

  def test_multi_document_search(self):
    # Search should work across multiple documents
    self.search_engine.add_document("mums_ab_organizational_structure.pdf")
    self.search_engine.add_document("mums_ab_product_descriptions.pdf")
    self.search_engine.add_document("mums_ab_annual_sales_report.pdf")

    org_results = self.search_engine.search("Who leads the company?")
    product_results = self.search_engine.search("What ingredients are used?")

    self.assertGreater(len(org_results), 0, "Should find org structure info")
    self.assertGreater(len(product_results), 0, "Should find product info")

  def test_result_contains_relevant_content(self):
    # Verify results actually contain relevant information
    self.search_engine.add_document("mums_ab_organizational_structure.pdf")
    results = self.search_engine.search("CEO")

    self.assertGreater(len(results), 0)
    combined_results = " ".join(results).lower()
    self.assertIn("ceo", combined_results, "Results should contain the queried term")

  def test_formulate_answer(self):
    # Formulate answer should be a string and contain "Anna Svensson"
    self.search_engine.add_document("mums_ab_organizational_structure.pdf")
    self.search_engine.add_document("mums_ab_product_descriptions.pdf")
    self.search_engine.add_document("mums_ab_annual_sales_report.pdf")

    results = self.search_engine.rag.formulate_answer("Who is the CEO?")
    results2 = self.search_engine.rag.formulate_answer("What is the total ammount of sales in January?")
    self.assertIn("Anna Svensson", results)
    self.assertIsInstance(results, str)
    print(f"results: {results2}")


  