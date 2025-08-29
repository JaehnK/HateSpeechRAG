import pandas as pd

from pprint import pprint
from LangChainService import *
from YouTubeDao import *

def main() -> int:
    # Set up the database and create tables
    youtube = YouTubeDBSetup()
    youtube.create_tables()
    pprint("Youtube DB Initalise Finished")
    pprint(youtube.get_data_count())

    # unique_channel_ids = youtube.get_unique_channel_ids()
    # pprint(unique_channel_ids)

    # for channel_id in unique_channel_ids:
    #     channel_comments = youtube.get_comments_by_video_id(channel_id)
    #     print(f"Comments for channel ID {channel_id}:")
    #     pprint(channel_comments)
    return 0

if __name__ == "__main__":

    # Run the main function
    main()