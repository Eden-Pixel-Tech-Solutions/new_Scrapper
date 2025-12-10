let timer = null;

document.getElementById("searchBox").addEventListener("keyup", function () {
    clearTimeout(timer);
    const query = this.value.trim();

    if (query.length < 2) {
        document.getElementById("resultBox").innerHTML = "";
        return;
    }

    timer = setTimeout(() => runSearch(query), 300); // debounce
});

function runSearch(query) {
    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query: query})
    })
    .then(res => res.json())
    .then(data => renderResults(data))
    .catch(err => console.error(err));
}

function renderResults(data) {
    if (data.error) {
        document.getElementById("resultBox").innerHTML =
            `<div class="alert alert-danger">${data.error}</div>`;
        return;
    }

    const best = data.best_match;

    document.getElementById("resultBox").innerHTML = `
        <div class="card shadow-sm p-3">
            <h5>Engine Used: ${data.engine}</h5>
            <h4 class="mt-2">${best?.title || "No Match"}</h4>
            <p><strong>Product Code:</strong> ${best?.product_code || "-"}</p>
            <p><strong>Category:</strong> ${data.detected_category}</p>
            <p><strong>Relevancy Score:</strong> ${data.relevancy_score.toFixed(4)}</p>
            <hr>
            <h6>Top Matches</h6>
            <ul>
                ${data.top_matches.map(t => `
                    <li>
                        <strong>${t.title}</strong> (${t.product_code}) — Score: ${t.relevancy.toFixed(4)}
                    </li>
                `).join("")}
            </ul>
        </div>
    `;
}
