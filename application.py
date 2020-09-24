from flask import Flask, render_template, request
import pytextrank
import spacy
import nltk
import re
import heapq

app = Flask(__name__)

def phrase_rank(text, count):
    nlp = spacy.load("en_core_web_sm")

    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

    doc = nlp(text)

    return doc._.phrases[:count]

def sentence_rank(text):
    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    sentence_list = nltk.sent_tokenize(article_text)
    stopwords = nltk.corpus.stopwords.words('indonesian')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    
    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)
    
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(20, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)

    return summary_sentences

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/summary', methods=["POST"])
def summary():
    original_text = request.form.get("original_text")
    keyword_count = int(request.form.get("keyword_count"))
    keywords = phrase_rank(original_text, keyword_count)
    main_points = sentence_rank(original_text)
    return render_template("summary.html", original_text=original_text, keywords=keywords, main_points=main_points)