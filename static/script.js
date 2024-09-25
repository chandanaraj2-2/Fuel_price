document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const dateInput = document.getElementById('date').value;
    const cityInput = document.getElementById('city').value;
    const fuelTypeInput = document.getElementById('fuelType').value;
    

    // Extract day, month, and year from the date input
    const date = new Date(dateInput);
    const day = date.getDate();
    const month = date.getMonth() + 1; // Months are zero-indexed
    const year = date.getFullYear();

    // Call the backend API to get the prediction
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            day: day,
            month: month,
            year: year,
            city: cityInput,
            fuelType: fuelTypeInput
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Predicted price: ${data.predicted_price.toFixed(2)}`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'Error predicting price.';
    });
});