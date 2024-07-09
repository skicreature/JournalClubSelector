import unittest
import tempfile
import shutil
import os
import pandas as pd
import json
import numpy as np
from journal_club_selector import JournalClubArticleSelector

class TestJournalClubArticleSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        
        # Create mock data files
        cls.landmark_articles_data = os.path.join(cls.test_dir, "landmark_articles_data.csv")
        cls.previous_presentations_file = os.path.join(cls.test_dir, "previous_presentations.txt")
        cls.embeddings_cache_file = os.path.join(cls.test_dir, "embeddings_cache.json")
        cls.priority_topics_file = os.path.join(cls.test_dir, "priority_topics.json")
        
        # Create mock landmark articles folder
        cls.landmark_articles_folder = os.path.join(cls.test_dir, "landmark_articles")
        os.mkdir(cls.landmark_articles_folder)
        
        # Create mock data
        pd.DataFrame({
            'id': ['1', '2', '3'],
            'title': ['Radiation Therapy Basics', 'Advanced IMRT Techniques', 'Proton Therapy in Pediatrics'],
            'authors': ['Author1', 'Author2', 'Author3'],
            'citation': ['Citation1', 'Citation2', 'Citation3'],
            'summary': ['Basic concepts of radiation therapy', 'Intensity-modulated radiation therapy advancements', 'Application of proton therapy in pediatric cancers'],
            'year': [2000, 2010, 2020],
            'citation_count': [100, 50, 10],
            'journal_impact_factor': [5, 3, 2]
        }).to_csv(cls.landmark_articles_data, index=False)
        
        with open(cls.previous_presentations_file, 'w') as f:
            f.write("Citation2\n")
        
        with open(cls.priority_topics_file, 'w') as f:
            json.dump({'radiation therapy': 3, 'IMRT': 2, 'proton therapy': 1}, f)
        
        # Create mock PDF files
        for i in range(1, 4):
            with open(os.path.join(cls.landmark_articles_folder, f"{i}.pdf"), 'w') as f:
                f.write(f"Mock content for article {i}")

    @classmethod
    def tearDownClass(cls):
        # Remove temporary directory and its contents
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        # Create a new selector instance for each test
        self.selector = JournalClubArticleSelector(
            landmark_articles_folder=self.landmark_articles_folder,
            landmark_articles_data=self.landmark_articles_data,
            previous_presentations_file=self.previous_presentations_file,
            voyageai_api_key="pa-HI4YNvR6YCQgpcrjNCQZSB4GS8MO1kpJbMr9-q3jEOQ",  # Replace with your actual API key
            embeddings_cache_file=self.embeddings_cache_file,
            priority_topics_file=self.priority_topics_file
        )

    def test_load_previous_presentations(self):
        self.assertEqual(self.selector.previous_presentations, ["Citation2"])

    def test_load_priority_topics(self):
        expected_topics = {'radiation therapy': 3, 'IMRT': 2, 'proton therapy': 1}
        self.assertEqual(self.selector.priority_topics, expected_topics)

    def test_generate_priority_topic_embeddings(self):
        self.assertEqual(len(self.selector.priority_topic_embeddings), 3)
        self.assertTrue(all(len(emb) == 1024 for emb in self.selector.priority_topic_embeddings.values()))

    def test_calculate_semantic_similarity(self):
        vec1 = np.random.rand(1024)
        vec2 = np.random.rand(1024)
        similarity = self.selector.calculate_semantic_similarity(vec1, vec2)
        self.assertGreaterEqual(similarity, -1)
        self.assertLessEqual(similarity, 1)

    def test_calculate_article_score(self):
        article = {
            'title': 'Test Article',
            'year': 2022,
            'citation_count': 500,
            'journal_impact_factor': 7
        }
        topic_embedding = np.random.rand(1024)
        article_embedding = np.random.rand(1024)
        score = self.selector.calculate_article_score(article, topic_embedding, article_embedding)
        self.assertGreater(score, 0)
        self.assertLess(score, 1)

    def test_find_relevant_landmark_articles(self):
        articles = self.selector.find_relevant_landmark_articles("radiation therapy techniques", 2)
        self.assertEqual(len(articles), 2)
        self.assertNotIn("Citation2", [a['citation'] for a in articles])
        for article in articles:
            self.assertIn('score', article)

    def test_find_recent_article(self):
        article = self.selector.find_recent_article("IMRT advancements")
        self.assertIsNotNone(article)
        self.assertIn('score', article)

    def test_select_articles(self):
        result = self.selector.select_articles("radiation therapy techniques")
        self.assertIn('landmark_articles', result)
        self.assertIn('recent_article', result)
        self.assertEqual(len(result['landmark_articles']), 2)
        self.assertIsNotNone(result['recent_article'])

    def test_analyze_coverage(self):
        coverage = self.selector.analyze_coverage()
        print("Coverage:", coverage)
        print("Priority topics:", self.selector.priority_topics)
        print("Previous presentations:", self.selector.previous_presentations)
        self.assertIn('covered_topics', coverage)
        self.assertIn('uncovered_topics', coverage)
        self.assertIn('IMRT', coverage['covered_topics'])  # 'Citation2' is about IMRT
        self.assertIn('radiation therapy', coverage['uncovered_topics'])  # Should not be covered by IMRT paper
        self.assertIn('proton therapy', coverage['uncovered_topics'])  # Should not be covered by IMRT paper

    def test_get_article_embeddings(self):
        file_path = os.path.join(self.landmark_articles_folder, "1.pdf")
        embedding = self.selector.get_article_embeddings(file_path)
        self.assertEqual(len(embedding), 1024)
        # Test caching
        self.assertIn(os.path.basename(file_path), self.selector.embeddings_cache)

    def test_update_embeddings(self):
        self.selector.update_embeddings()
        self.assertEqual(len(self.selector.embeddings_cache), 3)  # We have 3 mock articles

if __name__ == '__main__':
    unittest.main()