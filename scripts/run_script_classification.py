import os
import sys
from pprint import pprint

from dotenv import load_dotenv


def _add_project_root_to_syspath():
    """Ensure project root is on sys.path so `from src...` imports work."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root


def main():
    project_root = _add_project_root_to_syspath()
    load_dotenv()

    from src.dao import YouTubeDBSetup, VectorStoreDao
    from src.embedding import EmbeddingModelFactory
    from src.llm import HateSpeechRAGChain
    from src.service.YouTubeScriptClassifier import YouTubeScriptClassifier
    
    # Environment-configurable settings
    persist_dir = os.path.abspath(os.path.join(project_root, "data/vectorstores/hate_speech_vectorstore"))
    base_script_dir = os.getenv("YOUTUBE_SCRIPT_BASE_DIR")
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model_name = os.getenv("LLM_MODEL_NAME")
    retriever_k = int(os.getenv("RETRIEVER_K", "5"))

    # Initialize DAOs and RAG chain
    youtube_dao = YouTubeDBSetup()

    vectorstore_dao = VectorStoreDao(
        persist_directory=persist_dir,
        embedding_model=EmbeddingModelFactory.create_embedding_model("upstage"),
        collection_name="hate_speech_collection",
    )
    vectorstore_dao.create_vector_store()
    vectorstore_dao.initialize_retriever(retriever_type="basic", k=retriever_k)

    rag_chain = HateSpeechRAGChain(
        dao=vectorstore_dao,
        llm=,
        model_name='gpt-5',
    )

    classifier = YouTubeScriptClassifier(youtube_dao=youtube_dao, rag_chain=rag_chain)

    # Fetch all distinct video IDs via YouTubeDBSetup
    video_ids = youtube_dao.get_unique_video_ids() or []
    print(f"Found {len(video_ids)} unique video IDs to process")

    any_error = False
    for video_id in video_ids:
        print(f"Processing video_id: {video_id}")
        try:
            result = classifier.classify_video_script(video_id=video_id, 
                                        absolute_dir="/home/jaehun/lab/YouTubeHateSpeech")
            total = result.get("total_sentences", 0)
            classifications = result.get("classifications", [])
            successful = sum(1 for c in classifications if c and c.get("classification_result"))
            failed = total - successful
            pprint({
                "video_id": video_id,
                "total_sentences": total,
                "saved_success": successful,
                "failed": failed,
            })
        except Exception as e:
            any_error = True
            print(f"❌ Error processing {video_id}: {e}")

    if any_error:
        sys.exit(1)


if __name__ == "__main__":
    main()


