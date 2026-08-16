from dotenv import load_dotenv

import kagglehub

load_dotenv()

# Download latest version
path = kagglehub.competition_download('jigsaw-toxic-comment-classification-challenge')

print("Path to competition files:", path)

# /Users/st201n/.cache/kagglehub/competitions/jigsaw-toxic-comment-classification-challenge