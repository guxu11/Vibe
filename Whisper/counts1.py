# Created by guxu at 9/24/24
import re
text = '''
Student summary of the paper: T. Futterer et al.:“ChatGPT in education: global reactions to AI innovations”, Nature September 2023
Research Background
ChatGPT was released on November 30, 2022, which significantly influenced lots of areas. The paper discussed people's attitude towards ChatGPT’s effect on education by collecting and analyzing post/tweets on Twitter, with the aim to figure out (1) the most prevalent topics discussed regarding ChatGPT in education and (b) how users discussed these topics over this initial implementation period.
Theoretical background
ChatGPT is the newest model among GPT models, released by OpenAI. It can generate human-style natural languages. ChatGPT was also fine-tuned using RLHF to reduce the risk of generating incorrect or harmful content and make more appropriate answers. 
This paper also revealed the opportunities and risks of applying ChatGPT to education. ChatGPT and other LLM can help personalized and adaptive learning, and help visually impaired students or those who have difficulties in reading. Besides, ChatGPT is reactive, with the ability to give the feedback of learning. On the other hand, they have the risk of generating plausible-sounding but nonsense words because there is not enough ground truth during training. What’s more, ChatGPT itself won’t help students foster critical thinking, which needs educators to give guidance to make the best use of it.
People usually have different reactions to ground-breaking innovations, their positive attitudes are crucial to the wide-spread application of these technologies. People’s attitudes towards ChatGPT are complicated based on a research from Twitter, where 52% of the tweets are positive while 32% are negative. This reflects some people worrying about ChatGPT can lead to students’ dependency on it when writing and thinking. On the contrary, other people are happy to see the potential of revolution in education.
Twitter can provide large-scale and extendable public perception data, which make it an ideal data source of research on technical innovations. So this paper applied Twitter data on this research to answer three main questions:
(RQ1) What was the global reception on Twitter about ChatGPT?
(RQ2) What topics related to ChatGPT emerged in the education topical cluster during this initial period on Twitter?
(RQ3) How were the most prevalent educational topics related to ChatGPT discussed on Twitter?
Research Methods
The research team collected 16,743,036 tweets related to ChatGPT between 11/30/2022 and 01/31/2023. After some necessary filters, they finally got 16,743,036 tweets, 5,537,942 users and 125,151 conversations.
This paper is utilizing Topic modeling and Sentiment analysis as the main research methods. For Topic modeling, this research used BERTopic to extract topics from the dataset. BERTopic uses document embedding (BERTweet and MiniLM models) to recognize similar tweets. And they used VADER to do the Sentiment analysis, which shows great accuracy while processing Tweets data.
Results
RQ1: Figs 1A and 1B show a rapid increase in the number of tweets. Tweets related to ChatGPT increased from 0 before 11/30/2022 to over 550k in Jan 2023, which illustrates ChatGPT raised large-scale discussions. Beside, Fig 2 shows people around the world are discussing ChatGPT, mostly in English-speaking countries. What’s more, tweets from the first several days are almost positive but the distribution of attitudes became balanced over time, reaching positive-negative-neutral = 40-30-30. To conclude, the number of related tweets are significantly rising and the sentiments are becoming diverse and rational as time goes.
RQ2: This paper got 128 topics related to ChatGPT, and education was the third most popular topic among them. They also do further research on 10 education-related topics. The most popular topic is the chance, restriction and result of applying ChatGPT to education. The second one is student cheating using ChatGPT. Other education topics contain ChatGPT in academia, ban ChatGPT in educational organizations, etc. Besides, they also analyzed the relationship between these educational topics, with a result the ten educational topics are closely related.
RQ3: The research demonstrates ChatGPT-related tweets are mainly positive in the beginning but the negative and neural tweets increased after 01/05. The most positive educational topic is the one about AI can decrease the cost of education and the most negative one is the capability of ChatGPT to replicate students’ papers. 
Discussion and Conclusion
ChatGPT raised wide-spread discussions among the world, especially in education. The number of tweets boosted in two months, indicating this technology has great influence like the Internet and Computer. People can learn and discuss GenAI tools rapidly on Twitter, allowing educators more time to discuss the potential usage of ChatGPT. Besides, education is the third most popular topic, which shows ChatGPT plays a very important role in education though more explorations are needed. 
In conclusion, ChatGPT raised complicated sentimental reactions among people, from positive innovation and exploration to deep thinking about the challenges it may bring. The educators need deeper and more specific discussion to have a complete comprehension about the function of ChatGPT in education.
'''

text = text.strip()
result = re.split(r'[\s]+', text)
print(result.__len__())
