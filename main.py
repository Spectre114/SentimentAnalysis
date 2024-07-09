from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from utils import process_tweet, build_freqs
import nltk
from nltk.corpus import twitter_samples
from os import getcwd
import math

nltk.download('twitter_samples')
nltk.download('stopwords')

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)),
                    np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)),
                   np.zeros((len(test_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)


def sigmoid(z):
    z = -z
    h = 1/(1 + np.exp(z))
    return h


def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        delta = h - y
        J = (-1/m) * (np.sum(np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h))))
        theta = theta - (alpha/m) * (np.dot(x.T, delta))
    J = float(J)
    return J, theta


def extract_features(tweet, freqs, process_tweet=process_tweet):
    word_l = process_tweet(tweet)
    x = np.zeros(3)
    x[0] = 1
    for word in word_l:
        pair_pos = (word, 1.0)
        pair_neg = (word, 0.0)
        if (pair_pos in freqs):
            x[1] += freqs[pair_pos]
        if (pair_neg in freqs):
            x[2] += freqs[pair_neg]
    x = x[None, :]
    assert (x.shape == (1, 3))
    return x


X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

Y = train_y
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)


def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict/")
async def predict(request: Request, tweet: str = Form(...)):
    y_hat = predict_tweet(tweet, freqs, theta)
    sentiment = "Positive" if y_hat > 0.5 else "Negative"
    return {"tweet": tweet, "sentiment": sentiment, "score": float(y_hat)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
