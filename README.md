# Journal Club Article Selector
## Overview
The Journal Club Article Selector is a Python-based tool designed to assist in selecting relevant articles for medical journal clubs, specifically tailored for resident physicians in radiotherapy. This tool aims to streamline the process of choosing appropriate articles for discussion, ensuring a balance between classic landmark papers and recent, potentially impactful research.

## Features

1. **Topic-based Article Selection**:
   - Selects articles based on user-provided topics.
   - Uses natural language processing (via VoyageAI) to match topics with article content.

2. **Balanced Article Mix**:
   - Selects two classic landmark papers for foundational knowledge.
   - Identifies one recent article (published within the last 2-3 years) with potential future impact.

3. **Priority Topic Coverage**:
   - Maintains a list of priority topics essential for resident education.
   - Ensures comprehensive coverage of these topics over a 2-year cycle.

4. **Article Scoring System**:
   - Calculates relevance scores based on multiple factors:
     - Topic similarity
     - Topic priority
     - Article age
     - Citation count
     - Journal impact factor

5. **Previous Presentation Tracking**:
   - Keeps record of previously presented articles to avoid repetition.

6. **Coverage Analysis**:
   - Provides insights into which priority topics have been covered and which need attention.

7. **Embedding Cache**:
   - Stores and reuses article embeddings to improve performance and reduce API calls.

## How It Works

1. **Initialization**:
   - Loads landmark articles, priority topics, and previous presentations data.
   - Initializes connection with VoyageAI for embedding generation.

2. **Article Selection Process**:
   a. User inputs a topic of interest.
   b. The system generates an embedding for the input topic.
   c. Landmark articles are scored based on similarity to the topic and other factors.
   d. Recent articles are searched and scored similarly.
   e. The top two landmark articles and the top recent article are selected.

3. **Scoring Mechanism**:
   - Calculates a composite score considering:
     - Relevance to the input topic (40% weight)
     - Priority of the topic (20% weight)
     - Article age (recent landmarks and true classics get slight boosts)
     - Citation count (15% weight)
     - Journal impact factor (15% weight)

4. **Coverage Analysis**:
   - Tracks which priority topics have been covered in previous sessions.
   - Identifies gaps in topic coverage to guide future selections.

## Usage

```python
selector = JournalClubArticleSelector(
    landmark_articles_folder='path/to/articles',
    landmark_articles_data='path/to/metadata.csv',
    previous_presentations_file='path/to/history.txt',
    voyageai_api_key='your_api_key',
    embeddings_cache_file='path/to/cache.json',
    priority_topics_file='path/to/priorities.json'
)

results = selector.select_articles("Stereotactic body radiation therapy for lung cancer")
print(results)

coverage = selector.analyze_coverage()
print(coverage)
```

## Requirements
- Python 3.9+
- pandas
- numpy
- voyageai
- scholarly

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your VoyageAI API key

## Testing
Run the test suite using pytest:
```
pytest test_journal_club_selector.py
```

## Future Enhancements
- Web interface for easier interaction
- Integration with literature databases for more comprehensive article search
- Machine learning model to predict future landmark papers

## Contributors
[List of contributors]

## License
[Specify the license]