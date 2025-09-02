from typing import List

class BaseYouTubeClassifier:
    def __init__(self, youtube_dao, rag_chain):
        self.youtube_dao = youtube_dao
        self.connection = self.youtube_dao.get_connection()
        self.cursor = self.connection.cursor()
        self.rag_chain = rag_chain

    def _classify_sentences(self, sentences: List[str]):
        pass

    def _classify_sentences_async(self, sentences: List[str]):
        pass
