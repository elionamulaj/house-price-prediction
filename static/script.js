const form = document.getElementById("prediction-form");

const button = document.getElementById("predict-button");

const result = document.getElementById("prediction-result");

const description = document.getElementById("result-description");

const errorMessage = document.getElementById("error-message");


// Format prices in US dollars
const currencyFormatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
});


// Load actual machine learning metrics
async function loadMetrics() {

    try {

        const response = await fetch("/api/metrics");

        if (!response.ok) {
            throw new Error("Unable to load metrics.");
        }

        const metrics = await response.json();

        document.getElementById("model-name")
            .textContent = metrics.model;

        document.getElementById("r2-score")
            .textContent = Number(metrics.r2).toFixed(3);

        document.getElementById("mae-score")
            .textContent = currencyFormatter.format(metrics.mae);

    } catch (error) {

        document.getElementById("model-name")
            .textContent = "Unavailable";

        console.error(error);

    }

}


// Handle form submission
form.addEventListener("submit", async function(event) {

    event.preventDefault();

    errorMessage.textContent = "";

    const data = {

        OverallQual: Number(
            document.getElementById("OverallQual").value
        ),

        GrLivArea: Number(
            document.getElementById("GrLivArea").value
        ),

        GarageCars: Number(
            document.getElementById("GarageCars").value
        ),

        TotalBsmtSF: Number(
            document.getElementById("TotalBsmtSF").value
        ),

        FullBath: Number(
            document.getElementById("FullBath").value
        ),

        YearBuilt: Number(
            document.getElementById("YearBuilt").value
        ),

        YearRemodAdd: Number(
            document.getElementById("YearRemodAdd").value
        ),

        BedroomAbvGr: Number(
            document.getElementById("BedroomAbvGr").value
        ),

        LotArea: Number(
            document.getElementById("LotArea").value
        ),

        Neighborhood:
            document.getElementById("Neighborhood").value,

        HouseStyle:
            document.getElementById("HouseStyle").value,

        ExterQual:
            document.getElementById("ExterQual").value

    };

    if (data.YearRemodAdd < data.YearBuilt) {

        errorMessage.textContent =
            "Remodel year cannot be earlier than the construction year.";

        return;

    }

    button.disabled = true;

    button.textContent = "Predicting...";

    result.textContent = "$ —";

    description.textContent = "Analyzing property details...";

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });

        const prediction = await response.json();

        if (!response.ok || !prediction.success) {

            throw new Error(
                prediction.error || "Prediction failed."
            );

        }

        result.textContent = currencyFormatter.format(
            prediction.predicted_price
        );

        description.textContent =
            `Estimated using ${prediction.model}.`;

    } catch (error) {

        errorMessage.textContent = error.message;

        description.textContent =
            "Unable to generate a prediction.";

    } finally {

        button.disabled = false;

        button.textContent = "Predict House Price";

    }

});


loadMetrics();