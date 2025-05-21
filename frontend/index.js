
let myChart = null

document.getElementById("airport-form").addEventListener("submit", async function (event) {
    event.preventDefault();
    var airport_code = document.getElementById("code").value;
    var hour = document.getElementById("hour").value
    var day = document.getElementById("day").value;
    var month = document.getElementById("month").value;
    var year = document.getElementById("year").value;
    var output = document.getElementById("output");
    var date = new Date();
    var reqdate = new Date(year, month - 1, day, 23, 59, 59);
    console.log("today date", date);
    console.log("req date", reqdate);
    console.log(airport_code, day, month, year);
    if (!Number.isInteger(parseInt(day)) ||  !Number.isInteger(parseInt(month)) ||  !Number.isInteger(parseInt(year))) {
        output.innerText = "Please enter integers";
    }
    else if (day < 0 || day > 31 || month < 0 || month > 12 || year < 2000 || hour < 0 || hour > 24) {
        output.innerText = "Please enter a valid date"
    } else if ( reqdate < date ) {
        output.innerText = "Your date is in the past"

    }
    else
    {
        const prediction = await getPrediction(airport_code, hour, day, month, year)
        const graphInfo = await getGraphinfo(airport_code, hour, day, month, year)
        console.log("function called: ", prediction, graphInfo);
        output.innerText = "The estimated wait time at TSA is " + prediction + " minutes."

        const ctx = document.getElementById('myChart');
        if (myChart !== null) {
            myChart.clear()
            myChart.destroy();
        }

        myChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Wait Times',
                    data: graphInfo,
                    borderColor: 'black',

                    tension: 0.35,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'TSA Wait Times',
                        font: {
                            size: 20
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Military Time',
                            font: {
                                size: 14
                            }
                        },
                        ticks: {
                            stepSize: 1
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Wait Time (mins)',
                            font: {
                                size: 14
                            }
                        }
                    }
                }
            }
        });

    }
    console.log("function called");


});
const infostart = document.getElementById("inf");
const infobox = document.getElementById("info");
infobox.hidden = true;
infostart.addEventListener('mouseover', ()=>{
    infobox.hidden = false;
});
infostart.addEventListener('mouseout', ()=>{
    infobox.hidden = true;
});


async function getPrediction(airport_code, hour, day, month, year) {
    return fetch("http://127.0.0.1:5001/predict", {
        method:"POST",
        body: JSON.stringify({
            air_code: airport_code,
            HourRange: hour,
            day: day,
            month: month,
            year: year,

        }),
        headers: {
        "Content-type": "application/json; charset=UTF-8"
        }

    })
        .then((response)=>{
            return response.json();
        })


}

async function getGraphinfo(airport_code, hour, day, month, year) {
    return fetch("http://127.0.0.1:5001/graph", {
        method:"POST",
        body: JSON.stringify({
            air_code: airport_code,
            HourRange: hour,
            day: day,
            month: month,
            year: year,

        }),
        headers: {
        "Content-type": "application/json; charset=UTF-8"
        }

    })
        .then((response)=>{
            return response.json();
        })


}
