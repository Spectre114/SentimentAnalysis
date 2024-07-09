document
  .getElementById("tweet-form")
  .addEventListener("submit", async function (event) {
    event.preventDefault();
    const tweet = document.getElementById("tweet").value;
    const formData = new FormData();
    formData.append("tweet", tweet);

    const response = await fetch("/predict/", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    document.getElementById("result").innerHTML = `<strong>Tweet:</strong> ${
      result.tweet
    }<br><strong>Sentiment:</strong>${result.sentiment} ${
      result.sentiment === "Positive" ? "😀😀😀" : "😔😔😔"
    }`;
  });
