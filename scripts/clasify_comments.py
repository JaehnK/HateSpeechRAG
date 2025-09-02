import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.service import YouTubeContentClassifier


def main():
    classifier = YouTubeContentClassifier()

if __name__ == "__main__":
    main()