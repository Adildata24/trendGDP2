<script>
    // Function to parse the interest data and generate the chart
    function generateChart(interestData, keywords) {
        const labels = interestData.index;
        const datasets = [];

        for (const keyword of keywords) {
            datasets.push({
                label: keyword,
                data: interestData[keyword],
                fill: false,
                borderColor: getRandomColor(),
                borderWidth: 2,
            });
        }

        const ctx = document.getElementById('interestChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'month',
                            displayFormats: {
                                month: 'MMM YYYY',
                            },
                        },
                    },
                    y: {
                        beginAtZero: true,
                        suggestedMax: 100,
                    },
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                    },
                },
            },
        });
    }

    // Function to generate a random color for each keyword
    function getRandomColor() {
        const letters = '0123456789ABCDEF';
        let color = '#';
        for (let i = 0; i < 6; i++) {
            color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

    // Function to handle form submission and fetch the chart data
    document.querySelector('form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(event.target);
        const keywords = formData.get('keywords').split(',').map(keyword => keyword.trim());

        // Replace this URL with your server endpoint to fetch the interest data
        const endpoint = '/api/interest-data';
        const response = await fetch(endpoint, {
            method: 'POST',
            body: JSON.stringify(keywords),
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            alert('Failed to fetch interest data. Please try again.');
            return;
        }

        const data = await response.json();
        generateChart(data, keywords);
    });
</script>
