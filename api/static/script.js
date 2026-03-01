let gaugeChart;
let trendChart;
let simulationInterval = null;

// 🔊 Alert sound
const alertSound = new Audio("https://www.soundjay.com/buttons/beep-01a.mp3");

/* =========================
   DARK MODE TOGGLE
========================= */
document.getElementById("darkToggle").addEventListener("click", () => {
    document.body.classList.toggle("dark-mode");
});

/* =========================
   MAIN START
========================= */
async function startAnalysis() {

    const engineId = document.getElementById("engineId").value;

    if (!engineId) {
        alert("Please enter Engine ID");
        return;
    }

    showLoading(true);

    await runPrediction(engineId);
    await loadTrend(engineId);

    showLoading(false);

    startSimulation(engineId);
}

/* =========================
   LOADING SKELETON
========================= */
function showLoading(show) {
    const loader = document.getElementById("loadingSkeleton");
    loader.style.display = show ? "block" : "none";
}

/* =========================
   PREDICTION
========================= */
async function runPrediction(engineId) {

    const response = await fetch("/predict_by_engine", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({engine_id: parseInt(engineId)})
    });

    const data = await response.json();

    if (data.error) {
        alert(data.error);
        return;
    }

    updateGauge(data.failure_probability);
    updateStatus(data.predicted_class, data.failure_probability);
}

/* =========================
   ANIMATED GAUGE
========================= */
function updateGauge(prob) {

    const ctx = document.getElementById("gaugeChart").getContext("2d");
    const gaugeValue = document.getElementById("gaugeValue");

    if (gaugeChart) gaugeChart.destroy();

    const percentage = prob * 100;

    let color;

    if (percentage < 40) {
        color = "#16a34a";
    } 
    else if (percentage < 70) {
        color = "#f59e0b";
    } 
    else {
        color = "#dc2626";
    }

    gaugeChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            datasets: [{
                data: [percentage, 100 - percentage],
                backgroundColor: [color, "#e5e7eb"],
                borderWidth: 0
            }]
        },
        options: {
            rotation: -90,
            circumference: 180,
            cutout: "80%",
            animation: { duration: 1500 },
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });

    // Smooth number animation
    let current = 0;
    const steps = 50;
    const increment = percentage / steps;
    let count = 0;

    const interval = setInterval(() => {
        current += increment;
        count++;

        if (count >= steps) {
            current = percentage;
            clearInterval(interval);
        }

        gaugeValue.innerText = current.toFixed(1) + "%";

    }, 1500 / steps);
}

/* =========================
   STATUS + RISK BADGE
========================= */
function updateStatus(prediction, prob) {

    const statusText = document.getElementById("statusText");
    const probability = document.getElementById("probability");
    const badge = document.getElementById("riskBadge");
    const resultCard = document.querySelector(".result-card");

    const percent = prob * 100;
    probability.innerText = "Failure Probability: " + percent.toFixed(2) + "%";

    resultCard.classList.remove("danger-pulse");

    if (percent < 40) {
        badge.innerText = "LOW RISK";
        badge.style.background = "#16a34a";
    }
    else if (percent < 70) {
        badge.innerText = "MEDIUM RISK";
        badge.style.background = "#f59e0b";
    }
    else {
        badge.innerText = "HIGH RISK";
        badge.style.background = "#dc2626";
        resultCard.classList.add("danger-pulse");
        alertSound.play(); // 🔊 sound alert
    }

    if (prediction === 1) {
        statusText.innerHTML =
            "<i class='fa-solid fa-triangle-exclamation'></i> Failure Likely";
        statusText.style.color = "#dc2626";
    } else {
        statusText.innerHTML =
            "<i class='fa-solid fa-circle-check'></i> Machine Healthy";
        statusText.style.color = "#16a34a";
    }
}

/* =========================
   SENSOR TREND GRAPH
========================= */
async function loadTrend(engineId) {

    const sensor = document.getElementById("sensorSelect").value;

    const response = await fetch(`/engine_trend/${engineId}/${sensor}`);
    const data = await response.json();

    if (data.error) return;

    const ctx = document.getElementById("trendChart").getContext("2d");

    if (trendChart) trendChart.destroy();

    trendChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.cycle,
            datasets: [{
                label: sensor + " Trend",
                data: data.values,
                borderColor: "#0ea5e9",
                backgroundColor: "rgba(14,165,233,0.15)",
                tension: 0.35,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            animation: { duration: 1200 },
            scales: {
                x: { title: { display: true, text: "Cycle" } },
                y: { title: { display: true, text: "Sensor Value" } }
            }
        }
    });
}

/* =========================
   REAL-TIME SIMULATION
========================= */
function startSimulation(engineId) {

    if (simulationInterval)
        clearInterval(simulationInterval);

    simulationInterval = setInterval(async () => {
        await runPrediction(engineId);
    }, 5000);
}

/* =========================
   PDF EXPORT
========================= */
async function downloadReport() {

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();

    const engineId = document.getElementById("engineId").value;
    const status = document.getElementById("statusText").innerText;
    const probability = document.getElementById("probability").innerText;
    const sensor = document.getElementById("sensorSelect").value;

    doc.setFontSize(18);
    doc.text("Predictive Maintenance Report", 20, 20);

    doc.setFontSize(12);
    doc.text("Engine ID: " + engineId, 20, 40);
    doc.text("Selected Sensor: " + sensor, 20, 50);
    doc.text(status, 20, 60);
    doc.text(probability, 20, 70);

    doc.save("Engine_Report_" + engineId + ".pdf");
}