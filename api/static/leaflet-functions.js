let clusteredPoints = [];

const map = L.map('map').setView([43.45, -3.79], 10);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

function createCustomMarkerIcon(clusterNumber) {
    const iconsFolder = '/markers/';

    let markerIndex = 1;
    if (clusterNumber !== undefined) {
        markerIndex = clusterNumber + 1;
    }

    const iconUrl = iconsFolder + "marker-" + markerIndex.toString().padStart(2, '0') + ".png";
    return L.icon({
        iconUrl: iconUrl,
        iconSize: [40, 48],
        iconAnchor: [15, 42]
    });
}