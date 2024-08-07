# NymAI: Entity Based Sentiment Analyzer

## Project Overview
NymAI is an advanced sentiment analysis tool developed under the team name "Alzcur" by Gökdeniz Kuruca. This project was meticulously crafted within a span of one month to participate in the Teknofest 2024 Turkish Natural Language Processing (NLP) competition.

The primary goal of this project is to analyze sentiments associated with various entities within textual data, leveraging cutting-edge BERT-based models for improved accuracy and effectiveness. The solution is designed to handle complex NLP tasks by combining named entity recognition (NER) with sentiment analysis, thus providing valuable insights into the sentiments related to specific entities mentioned in texts.

### Key Features
- **Entity-Based Sentiment Analysis**: Utilizes BERT-based architecture to perform sentiment analysis specifically focused on entities extracted from text.
- **Advanced Named Entity Recognition**: Employs robust NER techniques to identify and categorize entities such as organizations, locations, and more.
- **Multi-Language Support**: Supports translation and text processing in both English and Turkish. It is compatible with other languages ​​and can be switched between all languages.
- **Customizable and Extendable**: Designed with flexibility to accommodate various sentiment classes and entity types.

### Architecture
- **Data Preprocessing**: Involves cleaning and normalizing text data, handling special characters, and translating text as needed.
- **Entity Extraction**: Identifies and processes entities using state-of-the-art NER techniques.
- **Sentiment Analysis**: Utilizes BERT-based models to analyze and classify sentiments associated with identified entities.
- **Output Formatting**: Provides structured output containing sentiment classifications and related entity information.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chyp3r/NymAI-Entity-Based-Sentiment-Analyzer
   cd your-repository-directory

2. **Install Dependencies**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    pip install -r requirements.txt

3. **Install Spacy Models**:
    ```bash
    python -c "from utils import ensure_en_core_web; ensure_en_core_web('en_core_web_sm')"
    python -c "from utils import ensure_en_core_web; ensure_en_core_web('en_core_web_lg')"

## Example Usage
- To see the sentiment analysis in action, run the teknofest_app.py file, which serves as an example application for demonstrating the functionality of the MainModel:

    ```bash
    python teknofest_app.py

- This script will start a FastAPI application. You can test the sentiment analysis by sending a POST request to the /predict/ endpoint with a JSON payload containing the text to analyze.

- Here’s how you can test it using curl:

    ```bash
    curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": Türkiye Teknoloji Takımı tarafında düzenlenen bu yarışma harika}'

- You can also run the post request via http://127.0.0.1:8000/docs

## Model Architecture
### Overview
The NymAI-Entity-Based-Sentiment-Analyzer utilizes an advanced BERT-based architecture for Entity-Based Sentiment Analysis (EBSA). Unlike traditional Aspect-Based Sentiment Analysis (ABSA), which focuses on specific aspects or features within text, EBSA emphasizes the sentiment associated with specific entities mentioned in the text, such as companies or products.

### Components
1.  **Text Preprocessing**
- **Text Cleaning**:
    - **URL Removal**: Eliminates URLs and other web links to prevent irrelevant information from affecting analysis.
    - **Special Character Removal**: Strips out special characters and punctuation that do not contribute to sentiment analysis.
    - **Whitespace Handling**: Normalizes spaces and removes unnecessary white spaces to clean up the text.
    - **Translation**: Translates non-English text to English to ensure consistency in sentiment analysis, using translation functions like translate_to_en.
    - **Normalization**: Standardizes text by addressing inconsistencies such as different forms of company names or jargon.

2.  **Named Entity Recognition (NER)**
- **Entity Extraction**:
    - **SpaCy Models**: Utilizes spaCy's en_core_web_sm and en_core_web_lg models to detect named entities within the text. These entities typically include names of organizations, products, and other relevant entities.
    - **Entity Normalization**: Converts identified entities to a consistent format. This involves removing prefixes (e.g., '@') and standardizing entity names.

3.  **Entity-Based Sentiment Analysis (EBSA)**
- **BERT-Based Model**:
    - **Contextual Understanding**: Leverages BERT (Bidirectional Encoder Representations from Transformers), which captures the context of words by considering both the left and right context in a sentence.
    - **Sentiment Classification**: The BERT-based model is fine-tuned to classify sentiments specific to each identified entity. Sentiments are typically classified into categories such as positive, negative, and neutral.
- **Sentiment Scoring**: 
    - Assigns sentiment scores to each entity based on the surrounding text and context provided by the BERT model.

4.  **Output Formatting**
- **Result Aggregation**:
    - **Entity-Sentiment Mapping**: Maps each identified entity to its sentiment score, providing a clear understanding of how different entities are perceived in the text.
- **Structured Output**: 
    - Formats the results into a structured dictionary format that includes a list of entities along with their associated sentiment classifications.

### Detailed Workflow
1. **Input Text**:
    - The analysis begins with raw text that may contain mentions of various entities such as companies or products.

2. **Preprocessing**:
- **Clean Text**:
    - Remove URLs and special characters.
    - Normalize spaces and translate non-English text to English if required.
- **Entity Extraction**:
    - Use spaCy’s models to detect and extract named entities from the text.
    - Normalize extracted entities to ensure consistency.

3. **Sentiment Analysis**:
    - **Feed Processed Text**: Provide the cleaned and preprocessed text along with the identified entities to the BERT-based EBSA model.
    - **Classify Sentiments**: The model evaluates the sentiment associated with each entity based on its contextual understanding.

4. **Formatting and Output**:
    - **Format Results**: Create a structured output that includes each entity and its sentiment score.
    - **Return Results**: Provide the final results in a dictionary format, listing entities and their corresponding sentiment classifications.

### Model Summary
- **Text Preprocessing**: Ensures that input text is clean, normalized, and ready for analysis.
- **Entity Extraction**: Identifies and normalizes entities using advanced spaCy models.
- **Entity-Based Sentiment Analysis**: Uses a BERT-based model to analyze and classify sentiment associated with each entity.
- **Output Formatting**: Presents results in a clear and structured format, detailing the sentiment for each entity.
- This architecture combines sophisticated text preprocessing with powerful BERT-based sentiment analysis to deliver accurate and actionable insights into how specific entities are perceived in the text.

## Datasets
The datasets used for training and evaluation of the NymAI-Entity-Based-Sentiment-Analyzer are sourced from SemEval14, SemEval15, and SemEval16. These datasets are widely used benchmarks in the field of sentiment analysis and contain annotated reviews from various domains. The datasets are provided in CSV format and are located in the dataset folder of the project.

1. **SemEval14**: Includes reviews from laptop and restaurant domains. Each review is annotated with aspects and corresponding sentiments.
    - **Laptop Reviews**:
        Number of reviews: 3845
        Number of aspects: 3045
    - **Restaurant Reviews**:
        Number of reviews: 3300
        Number of aspects: 3813

2. **SemEval15**: Extends the dataset with more reviews and includes additional domains such as hotels and devices.
    - **Hotel Reviews**:
        Number of reviews: 2000
        Number of aspects: 1692


3. **SemEval16**: Further extends the dataset with a focus on more granular aspect categories and additional sentiment classes.
    - **Restaurant Reviews**:
        Number of reviews: 2000
        Number of aspects: 1743

    Each dataset contains columns that typically include:
    
        text: The text of the review.
        aspect: The specific entity or feature being reviewed.
        sentiment: The sentiment expressed towards the aspect (e.g., positive, negative, neutral).

These datasets are crucial for training the BERT-based entity-based sentiment analysis model, allowing it to learn from a diverse set of reviews and accurately predict sentiments across different entities and domains.

During dataset experiments, these datasets were also translated into Turkish and used for training. However, the results from the Turkish datasets did not meet the desired performance levels. Consequently, it was decided to keep the datasets in their original language and translate the sentences to be processed instead. This approach ensures better accuracy and consistency in sentiment analysis.

## Customizing Named Entity Recognition (NER) for Different Entity Types
In the ner.py file, the default implementation of the NER system focuses on identifying organization names (labeled as ORG) from the text. If you need to extract different types of entities (such as PERSON for people, LOC for locations, etc.), you will need to modify the NER system to accommodate these changes.

Here is how you can customize the entity extraction to focus on different types of entities:

1. **Locate the Entity Extraction Code**: Find the lines in ner.py where entities are being extracted using the SpaCy models:

    ```bash
    companies_sm = [ent.text.replace("@", "") for ent in doc_sm.ents if ent.label_ == "ORG"]
    companies_lg = [ent.text.replace("@", "") for ent in doc_lg.ents if ent.label_ == "ORG"]

2. **Change the Entity Label**: Replace ORG with the appropriate label for the entity type you are interested in. SpaCy provides a variety of labels for different entity types. Here are some common labels you might use:

        PERSON: People, including fictional.
        NORP: Nationalities or religious or political groups.
        FAC: Buildings, airports, highways, bridges, etc.
        ORG: Companies, agencies, institutions, etc.
        GPE: Countries, cities, states.
        LOC: Non-GPE locations, mountain ranges, bodies of water.
        PRODUCT: Objects, vehicles, foods, etc. (not services).
        EVENT: Named hurricanes, battles, wars, sports events, etc.
        WORK_OF_ART: Titles of books, songs, etc.
        LAW: Named documents made into laws.
        LANGUAGE: Any named language.

3. **Example Modification**: To extract PERSON entities instead of ORG, you would change the lines as follows:

4. **Save the Changes**: After making the necessary changes, save the [ner.py](https://github.com/chyp3r/NymAI-Entity-Based-Sentiment-Analyzer/blob/main/ner.py) file.

5. **Update Other Components**: Ensure that any other parts of your code that depend on these entities are appropriately updated to handle the new entity type.

    By following these steps, you can customize the entity extraction process to focus on the specific types of entities relevant to your application. This flexibility allows you to adapt the NER system to a wide range of use cases beyond the default focus on organizations.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/chyp3r/NymAI-Entity-Based-Sentiment-Analyzer?tab=Apache-2.0-1-ov-file) file for details.

## Citation
If you use this code or any part of it in your research, please consider citing our work:
    
    @misc{nymAI2024,
    author = {Gökdeniz Kuruca},
    title = {NymAI-Entity-Based-Sentiment-Analyzer},
    year = {2024},
    howpublished = {\url{https://github.com/chyp3r/NymAI-Entity-Based-Sentiment-Analyzer}},
    note = {Developed under the Alzcur team for the Teknofest 2024 Turkish Natural Language Processing competition.}
    }


## Acknowledgements

Parts of the code for this project were adapted from the [Aspect-Based Sentiment Analysis](https://github.com/ScalaConsultants/Aspect-Based-Sentiment-Analysis) repository. We extend our gratitude to the authors of this repository for their contributions, which were instrumental in the development of this project.

Please refer to the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) for details on the usage and distribution terms.


    
