Logger.clear()


function searchWorstAndBestCases() {

    var objective_metric_list = ["AP@k"];
    var optim_algo_list = ["BFGS", "CG", "Newton-CG", "SLSQP", "TNC", "dogleg", "nelder-mead", "powell", "trust-ncg", "COBYLA"];
    var we_model_list = ["models/GoogleNews-vectors-negative300.magnitude", "models/glove.840B.300d.magnitude"];
    var ground_truth_name_list = ["usa_states", "universe_solar_planets", "king_of_rome", "period_7_element"];

    var row_i = 3;
    var total = [objective_metric_list, optim_algo_list, we_model_list, ground_truth_name_list]
        .reduce(function (prev, curr) {
            return ((prev && prev.length) || prev) * curr.length
        })
    Logger.info("total: " + total);
    getSheet(MySettings.sheetWorstAndBestCasesName).getRange("A1").setValue("Wait.. -" + total);

    optim_algo_list.forEach(function (optim_algo) {
        objective_metric_list.forEach(function (objective_metric) {
            we_model_list.forEach(function (we_model) {
                ground_truth_name_list.forEach(function (ground_truth_name) {
                    var filters = {
                        objective_metric: objective_metric,
                        optim_algo: optim_algo,
                        we_model: we_model,
                        ground_truth_name: ground_truth_name,
                    }
                    var rwawabc = addRowWithAverageWorstAndBestCase(MySettings.sheetWorstAndBestCasesName, filters, row_i);
                    if (rwawabc) { // if something has been added...
                        row_i = row_i + 1;
                    }
                    total = total - 1;
                    getSheet(MySettings.sheetWorstAndBestCasesName).getRange("A1").setValue("Wait.. -" + total);
                });
            });
        });
    });
}

var cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
function addRowWithAverageWorstAndBestCase(sheetName, filters, row_i) {

    var col_i = 0;
    var filteredRowsObj = getFilteredRowsByFilters(filters);
    var allRows = filteredRowsObj.allRows;
    var filteredRows = filteredRowsObj.filteredRows;
    Logger.log('filters: ' + JSON.stringify(filters))
    Logger.log("filteredRows.length: " + filteredRows.length)
    if (filteredRows.length <= 0) { return false; }

    var worstCaseExperimentId = getBestAndWorstCaseExperimentIdFrom(allRows, filteredRows).worstCaseExperimentId;
    var bestCaseExperimentId = getBestAndWorstCaseExperimentIdFrom(allRows, filteredRows).bestCaseExperimentId;
    Logger.log("worstCaseExperimentId: " + worstCaseExperimentId)
    Logger.log("bestCaseExperimentId: " + bestCaseExperimentId)
    if (!worstCaseExperimentId || !bestCaseExperimentId) { return false; }

    var rowsOfWorstCaseExperiment = getRowByExperimentId(allRows, worstCaseExperimentId);
    var rowsOfBestCaseExperiment = getRowByExperimentId(allRows, bestCaseExperimentId);

    var worstCaseObj = getDataObjOfRowsTwin(rowsOfWorstCaseExperiment);
    var bestCaseObj = getDataObjOfRowsTwin(rowsOfBestCaseExperiment);

    // verify
    var rowCentroid = rowsOfWorstCaseExperiment["centroid"];
    if (filters.optim_algo !== rowCentroid[ColumnIndex.optim_algo]
        || filters.ground_truth_name !== rowCentroid[ColumnIndex.ground_truth_name]
        || filters.we_model !== rowCentroid[ColumnIndex.we_model]
        || filters.objective_metric !== rowCentroid[ColumnIndex.objective_metric]
    ) {
        Logger.log("ERROR: FILTER AND ROW VALUES MUST BE THE SAME!!!")
        Logger.log('filters: ' + JSON.stringify(filters))
        Logger.log('rowCentroid: ' + JSON.stringify(rowCentroid))
        Logger.log("ERROR --------------- fine.")
        getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue("ERROR");
        getSheet(sheetName).getRange(cols[col_i++] + row_i).setNote("ERROR: FILTER AND ROW VALUES MUST BE THE SAME!!! " + new Date());
        return true;
    }

    var today = new Date();
    var time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
    var dateTime = time;

    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(dateTime);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(filters.optim_algo);

    // worst case
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(worstCaseObj.centroidResult);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(worstCaseObj.optimizedResult);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(worstCaseObj.improvement);

    // best case
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(bestCaseObj.centroidResult);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(bestCaseObj.optimizedResult);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(bestCaseObj.improvement);


    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(filters.ground_truth_name);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(filters.we_model);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(filters.objective_metric);


    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(worstCaseObj.experimentId);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(bestCaseObj.experimentId);
    return true;

}


function getDataObjOfRowsTwin(rows) {
    var rowCentroid = rows["centroid"];
    var rowOptimized = rows["optimized"];
    var experimentId = rowCentroid[ColumnIndex.experimentId];
    var centroidResult = rowCentroid[ColumnIndex.objective_metric_result];
    var optimizedResult = rowOptimized[ColumnIndex.objective_metric_result];
    var improvement = optimizedResult / centroidResult - 1;
    return {
        centroidResult: centroidResult,
        optimizedResult: optimizedResult,
        improvement: improvement,
        experimentId: experimentId,
    }
}

// function onEdit(e) {
//     // e = {authMode=LIMITED, range=Range, source=Spreadsheet, oldValue=nelder-mead, user=emanuelemameli@gmail.com, value=BFGS}
//     var col = e.range.getColumn()
//     var row = e.range.getRow()
//     if (col === 1) {
//         if ([4, 6, 7].indexOf(row) > -1) {
//             // use case has changed
//             searchWorstAndBestCases()
//         }
//     }
// }

var ColumnIndex = function () { }
ColumnIndex.INFO = 7; // OPTIMIZED, CENTROID
ColumnIndex.optim_algo = 8; // BFGS, nelder-mead, powell.....
ColumnIndex.objective_metric = 9; // AP@k
ColumnIndex.objective_metric_result = 10; // 0.348842 ...
ColumnIndex.we_model = 36; //  models/GoogleNews-vectors-negative300.magnitude
ColumnIndex.ground_truth_name = 42; // usa_states
ColumnIndex.experimentId = 43;

var MySettings = function () { }
MySettings.sheetWorstAndBestCasesName = "Worst&BestCases";
MySettings.sheetResultsName = "Results";
MySettings.resultsRange = "A1:AS1000";

var filterChoosen = function (filters) {
    return function (row) {
        return row[ColumnIndex.INFO] === "OPTIMIZED"
            && row[ColumnIndex.optim_algo] === filters.optim_algo
            && row[ColumnIndex.objective_metric] === filters.objective_metric
            && row[ColumnIndex.we_model] === filters.we_model
            && row[ColumnIndex.ground_truth_name] === filters.ground_truth_name
    }
}


function getRowByExperimentId(rows, experimentId) {
    var rowsExperiment = rows.filter(function (row) {
        return row[ColumnIndex.experimentId] === experimentId
    })
    return {
        centroid: rowsExperiment.filter(function (row) {
            return row[ColumnIndex.INFO] === "CENTROID"
        })[0],
        optimized: rowsExperiment.filter(function (row) {
            return row[ColumnIndex.INFO] === "OPTIMIZED"
        })[0],
    }
}


function getBestAndWorstCaseExperimentIdFrom(allRows, filteredRows) {
    var worstCaseExperimentId = null;
    var bestCaseExperimentId = null;
    var worstCaseValue = null;
    var worstCaseCentroidValue = null;
    var bestCaseValue = null;

    for (var i = 0; i < filteredRows.length; i++) {
        const rowOptimized = filteredRows[i];

        // prendi l'id della riga e recupera anche la riga del CENTROID
        var experimentId = rowOptimized[ColumnIndex.experimentId];
        var rowCentroid = getRowByExperimentId(allRows, experimentId)["centroid"];

        // leggi objective_metric_result OPTIMIZED e sottrai a objective_metric_result CENTROID
        var objectiveMetricResultOptimized = rowOptimized[ColumnIndex.objective_metric_result];
        var objectiveMetricResultCentroid = rowCentroid[ColumnIndex.objective_metric_result];
        var actualDifference = objectiveMetricResultOptimized - objectiveMetricResultCentroid;

        // memorizza quella con l'improvement minore
        if (!worstCaseExperimentId
            || actualDifference < worstCaseValue
            || objectiveMetricResultCentroid < worstCaseCentroidValue
        ) {
            worstCaseExperimentId = experimentId;
            worstCaseValue = actualDifference;
        }

        // memorizza quella con l'improvement maggiore
        if (!bestCaseExperimentId
            || actualDifference > bestCaseValue
        ) {
            bestCaseExperimentId = experimentId;
            bestCaseValue = actualDifference;
        }
    }
    return {
        worstCaseExperimentId: worstCaseExperimentId,
        bestCaseExperimentId: bestCaseExperimentId,
    };
}




function getSheet(name) {
    var sheets = SpreadsheetApp.getActiveSpreadsheet().getSheets();
    for (var i = 0; i < sheets.length; i++) {
        if (sheets[i].getName() === name) {
            return sheets[i];
        }
    }
    return null;
}

var allRows = getSheet(MySettings.sheetResultsName).getRange(MySettings.resultsRange).getValues();
function getFilteredRowsByFilters(filters) {
    var filteredRows = allRows.getValues()
    filteredRows = filteredRows.filter(filterChoosen(filters))
    return {
        allRows: allRows,
        filteredRows: filteredRows
    };
}


