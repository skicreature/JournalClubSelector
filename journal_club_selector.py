import os
import pandas as pd
import json
from typing import List, Dict
import voyageai
import numpy as np
from datetime import datetime
from scholarly import scholarly
from collections import Counter

class JournalClubArticleSelector:
    def __init__(self, landmark_articles_folder: str, landmark_articles_data: str,
                 previous_presentations_file: str, voyageai_api_key: str,
                 embeddings_cache_file: str, priority_topics_file: str):
        self.landmark_articles_folder = landmark_articles_folder
        self.landmark_articles_data = pd.read_csv(landmark_articles_data)
        self.previous_presentations = self.load_previous_presentations(previous_presentations_file)
        self.voyageai_client = voyageai.Client(api_key=voyageai_api_key)
        self.embeddings_cache_file = embeddings_cache_file
        self.embeddings_cache = self.load_embeddings_cache()
        self.priority_topics = self.load_priority_topics(priority_topics_file)
        self.priority_topic_embeddings = self.generate_priority_topic_embeddings()

    def load_previous_presentations(self, file_path: str) -> List[str]:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def load_embeddings_cache(self) -> Dict[str, List[float]]:
        if os.path.exists(self.embeddings_cache_file):
            with open(self.embeddings_cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save_embeddings_cache(self):
        with open(self.embeddings_cache_file, 'w') as f:
            json.dump(self.embeddings_cache, f)

    def get_article_embeddings(self, file_path: str) -> List[float]:
        file_name = os.path.basename(file_path)
        if file_name in self.embeddings_cache:
            return self.embeddings_cache[file_name]

        with open(file_path, 'r') as f:
            content = f.read()
        embedding = self.voyageai_client.embed(content).embeddings[0]
        self.embeddings_cache[file_name] = embedding
        self.save_embeddings_cache()
        return embedding

    def load_priority_topics(self, file_path: str) -> Dict[str, int]:
        with open(file_path, 'r') as f:
            return json.load(f)

    def generate_priority_topic_embeddings(self) -> Dict[str, List[float]]:
        return {topic: self.voyageai_client.embed(topic).embeddings[0] for topic in self.priority_topics}

    def calculate_semantic_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def calculate_article_score(self, article: Dict, topic_embedding: List[float], article_embedding: List[float]) -> float:
        # Calculate similarity between topic and article
        article_similarity = self.calculate_semantic_similarity(topic_embedding, article_embedding)
        score = article_similarity * 0.4  # Base score from article similarity

        # Factor 1: Priority of the topic
        topic_priorities = [
            self.priority_topics[topic] * self.calculate_semantic_similarity(topic_embedding, pt_embedding)
            for topic, pt_embedding in self.priority_topic_embeddings.items()
        ]
        max_topic_priority = max(topic_priorities) if topic_priorities else 0
        score += max_topic_priority * 0.2

        # Factor 2: Article age (for landmark articles)
        if 'year' in article:
            current_year = datetime.now().year
            years_old = current_year - int(article['year'])
            if years_old <= 5:
                score += 0.1  # Boost for recent landmark articles
            elif years_old > 20:
                score += 0.05  # Small boost for true classics

        # Factor 3: Citation count (if available)
        if 'citation_count' in article:
            normalized_citations = min(article['citation_count'] / 1000, 1)  # Normalize to 0-1
            score += normalized_citations * 0.15

        # Factor 4: Journal impact factor (if available)
        if 'journal_impact_factor' in article:
            normalized_impact = min(article['journal_impact_factor'] / 10, 1)  # Normalize to 0-1
            score += normalized_impact * 0.15

        return score

    def find_relevant_landmark_articles(self, topic: str, num_articles: int) -> List[Dict]:
        topic_embedding = self.voyageai_client.embed(topic).embeddings[0]
        article_similarities = []

        for _, row in self.landmark_articles_data.iterrows():
            file_path = os.path.join(self.landmark_articles_folder, f"{row['id']}.pdf")
            if os.path.exists(file_path):
                article_embedding = self.get_article_embeddings(file_path)
                score = self.calculate_article_score(row, topic_embedding, article_embedding)
                article_similarities.append((score, row))

        article_similarities.sort(reverse=True, key=lambda x: x[0])
        relevant_articles = [
            {
                'title': article['title'],
                'authors': article['authors'],
                'citation': article['citation'],
                'summary': article['summary'],
                'score': score
            }
            for score, article in article_similarities
            if article['citation'] not in self.previous_presentations
        ][:num_articles]

        return relevant_articles

    def find_recent_article(self, topic: str) -> Dict:
        search_query = scholarly.search_pubs(topic)
        current_year = datetime.now().year
        recent_articles = []

        for article in search_query:
            pub_year = article['bib'].get('pub_year')
            if pub_year and int(pub_year) >= current_year - 3:
                article_dict = {
                    'title': article['bib'].get('title', ''),
                    'authors': ', '.join(article['bib'].get('author', [])),
                    'citation': f"{article['bib']['author'][0] if article['bib'].get('author') else 'Unknown'} et al., {article['bib'].get('venue', 'Unknown Venue')}, {pub_year}",
                    'summary': article['bib'].get('abstract', ''),
                    'year': int(pub_year),
                    'citation_count': article.get('num_citations', 0),
                }
                topic_embedding = self.voyageai_client.embed(topic, model="voyage-2").embeddings[0]
                article_embedding = self.voyageai_client.embed(article_dict['title'] + " " + article_dict['summary'], model="voyage-2").embeddings[0]
                score = self.calculate_article_score(article_dict, topic_embedding, article_embedding)
                recent_articles.append((score, article_dict))
                if len(recent_articles) >= 5:  # Limit to top 5 for efficiency
                    break

        recent_articles.sort(reverse=True, key=lambda x: x[0])
        if recent_articles:
            best_recent = recent_articles[0][1]
            best_recent['score'] = recent_articles[0][0]
            return best_recent
        return None

    def select_articles(self, topic: str) -> Dict:
        landmark_articles = self.find_relevant_landmark_articles(topic, 2)
        recent_article = self.find_recent_article(topic)

        return {
            'landmark_articles': landmark_articles,
            'recent_article': recent_article
        }

    def analyze_coverage(self) -> Dict:
        covered_topics = Counter()
        for citation in self.previous_presentations:
            article = self.landmark_articles_data[self.landmark_articles_data['citation'] == citation].iloc[0]
            article_embedding = self.get_article_embeddings(os.path.join(self.landmark_articles_folder, f"{article['id']}.pdf"))
            for topic, embedding in self.priority_topic_embeddings.items():
                similarity = self.calculate_semantic_similarity(article_embedding, embedding)
                print(f"Similarity between '{article['title']}' and topic '{topic}': {similarity}")  # Debug print
                if similarity > 0.7:  # Increased threshold
                    covered_topics[topic] += 1

        uncovered_topics = {topic: priority for topic, priority in self.priority_topics.items()
                            if covered_topics[topic] == 0}
        return {
            'covered_topics': dict(covered_topics),
            'uncovered_topics': uncovered_topics
        }

    def update_embeddings(self):
        for _, row in self.landmark_articles_data.iterrows():
            file_path = os.path.join(self.landmark_articles_folder, f"{row['id']}.pdf")
            if os.path.exists(file_path):
                self.get_article_embeddings(file_path)
        print("Embeddings updated and cached.")

# Usage example
if __name__ == "__main__":
    selector = JournalClubArticleSelector(
        landmark_articles_folder='path/to/landmark_articles',
        landmark_articles_data='path/to/landmark_articles_data.csv',
        previous_presentations_file='path/to/previous_presentations.txt',
        voyageai_api_key='your_voyageai_api_key',
        embeddings_cache_file='path/to/embeddings_cache.json',
        priority_topics_file='path/to/priority_topics.json'
    )

    topic = "Stereotactic body radiation therapy for lung cancer"
    selected_articles = selector.select_articles(topic)

    print("Landmark Articles:")
    for article in selected_articles['landmark_articles']:
        print(f"- {article['title']} (Score: {article['score']:.2f})")

    print("\nRecent Article:")
    if selected_articles['recent_article']:
        print(f"- {selected_articles['recent_article']['title']} (Score: {selected_articles['recent_article']['score']:.2f})")

    coverage_analysis = selector.analyze_coverage()
    print("\nTopic Coverage Analysis:")
    print(f"Covered topics: {coverage_analysis['covered_topics']}")
    print(f"Uncovered high-priority topics: {coverage_analysis['uncovered_topics']}")