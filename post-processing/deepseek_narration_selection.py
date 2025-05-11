import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

movie_subtitles_path = "Path of a txt file. The txt file is the movie subititle with timestamps. Movie subtitle can be downloaded from the website: https://www.subscene.co.in/"
with open(movie_subtitles_path, "r", encoding="utf-8") as file:
    movie_subtitles = file.read()

completion = client.chat.completions.create(
    model="deepseek-v3",
    messages=[
        {
            'role': 'user', 
            'content': 'You are an experienced trailer editor. Now you need to select appropriate sentences from the given txt file as the trailer\'s narration. Please read all the contents of the document carefully. Every three lines in the txt file are the subtitle index, the time when the subtitle appears, and the subtitle content. The following are suggestions for selecting subtitle sentences: First, the category of the movie corresponding to this trailer is: Action Epic, Sword & Scandal, Drama, and the subtitle sentence should be able to explain the theme of the movie; secondly, you should choose a complete and long sentence, or you can choose multiple sentences connected together; finally, choose a total of 10 sentences. The given document content is: {}'.format(movie_subtitles)
        }
    ]
)

print("Thinking process")
print(completion.choices[0].message.reasoning_content)

print("Final answer:")
print(completion.choices[0].message.content)