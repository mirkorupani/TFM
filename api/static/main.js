document.addEventListener('DOMContentLoaded', function() {
    axios.get('/points')
        .then(response => {
            loadPoints(response.data);
        })
        .catch(error => {
            console.error('Error fetching points:', error);
        });
});

let clusterData = null;

function loadPoints(points) {
    clusteredPoints = points;

    points.forEach(point => {
        let clusterNumber = point.cluster;
        let markerIcon = createCustomMarkerIcon(clusterNumber);
        const marker = L.marker([point.latitude, point.longitude], { icon: markerIcon });
        marker.addTo(map);
        marker.bindPopup(`<b>${point.name}</b><br>ID: ${point.id}<br>Cluster: ${point.cluster}<br>Click to train model`).openPopup();
        marker.on('click', () => {
            document.getElementById('output').innerHTML = '';
            document.getElementById('plot-ux').innerHTML = '';
            document.getElementById('plot-uy').innerHTML = '';
            document.getElementById('plot-waterlevel').innerHTML = '';
            document.getElementById('loading').style.display = 'block';

            axios.post('/train', { point: point, cluster_data: clusterData })
                .then(response => {
                    const result = response.data;
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('output').innerHTML = `
                        <h3 class="result-heading">Training Result for ${point.name}</h3>
                    `;
                    const fontSize = 10;
                    const titleMargin = 20;

                    Plotly.newPlot('plot-ux', [{
                        x: result.x,
                        y: result.y_u_x,
                        mode: 'lines',
                        type: 'scatter',
                        name: 'Actual'
                    }, {
                        x: result.x,
                        y: result.yPred_u_x,
                        mode: 'lines',
                        type: 'scatter',
                        name: 'Predicted'
                    }], {
                        title: { text: 'Actual vs Predicted Time Series (u_x)', font: { size: fontSize } },
                        margin: { t: titleMargin },
                        xaxis: { title: { text: 'Time', font: { size: fontSize } }, tickfont: { size: fontSize } },
                        yaxis: { title: { text: 'Current (m/s)', font: { size: fontSize } }, tickfont: { size: fontSize } },
                        legend: { font: { size: fontSize } }
                    });

                    Plotly.newPlot('plot-uy', [{
                        x: result.x,
                        y: result.y_u_y,
                        mode: 'lines',
                        type: 'scatter',
                        name: 'Actual'
                    }, {
                        x: result.x,
                        y: result.yPred_u_y,
                        mode: 'lines',
                        type: 'scatter',
                        name: 'Predicted'
                    }], {
                        title: { text: 'Actual vs Predicted Time Series (u_y)', font: { size: fontSize } },
                        margin: { t: titleMargin },
                        xaxis: { title: { text: 'Time', font: { size: fontSize } }, tickfont: { size: fontSize } },
                        yaxis: { title: { text: 'Current (m/s)', font: { size: fontSize } }, tickfont: { size: fontSize } },
                        legend: { font: { size: fontSize } }
                    });

                    Plotly.newPlot('plot-waterlevel', [{
                        x: result.x,
                        y: result.y_waterlevel,
                        mode: 'lines',
                        type: 'scatter',
                        name: 'Actual'
                    }, {
                        x: result.x,
                        y: result.yPred_waterlevel,
                        mode: 'lines',
                        type: 'scatter',
                        name: 'Predicted'
                    }], {
                        title: { text: 'Actual vs Predicted Time Series (Water Level)', font: { size: fontSize } },
                        margin: { t: titleMargin },
                        xaxis: { title: { text: 'Time', font: { size: fontSize } }, tickfont: { size: fontSize } },
                        yaxis: { title: { text: 'Water Level (m)', font: { size: fontSize } }, tickfont: { size: fontSize } },
                        legend: { font: { size: fontSize } }
                    });
                })
                .catch(error => {
                    console.error('Error training model:', error);
                });
        });
    });
}

document.getElementById('clusterize-button').addEventListener('click', () => {
    const numClusters = document.getElementById('num-clusters').value;
    document.getElementById('cluster-loading').style.display = 'block';
    axios.post('/cluster', { num_clusters: parseInt(numClusters) })
        .then(response => {
            document.getElementById('cluster-loading').style.display = 'none';
            const clusterId = response.data.cluster_id;
            // Store the cluster ID for later use
            localStorage.setItem('clusterId', clusterId);
            axios.get(`/clusters/${clusterId}`)
                .then(response => {
                    clusterData = response.data; // Store the cluster data for later use
                    loadPoints(response.data);
                })
                .catch(error => {
                    console.error('Error fetching cluster points:', error);
                });
        })
        .catch(error => {
            document.getElementById('cluster-loading').style.display = 'none';
            console.error('Error clustering points:', error);
        });
});