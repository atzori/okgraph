if (!Logger) {
    Logger = {}
    Logger.log = console.log
}
Logger.clear()


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
MySettings.resultsRange = "A1:AS5200";


function searchWorstAndBestCases() {

    var objective_metric_list = ["AP@k"];
    var optim_algo_list = ['powell', 'nelder-mead', 'BFGS', 'Newton-CG', 'CG', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'COBYLA'];
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
                    getSheet(MySettings.sheetWorstAndBestCasesName).getRange("A1").setValue("Wait.. -" + total + " [" + row_i + "]");
                });
            });
        });
    });
}

var cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']
function addRowWithAverageWorstAndBestCase(sheetName, filters, row_i) {

    var col_i = 0;
    var filteredRows = getFilteredRowsByFilters(filters);
    // Logger.log('filters: ' + JSON.stringify(filters))
    Logger.log("filteredRows.length: " + filteredRows.length)
    if (filteredRows.length <= 0) { return false; }

    var bestAndWorstCases = getBestAndWorstCaseFrom(filteredRows)
    var rowsOfWorstCaseExperiment = bestAndWorstCases.rowsOfWorstCaseExperiment;
    var rowsOfBestCaseExperiment = bestAndWorstCases.rowsOfBestCaseExperiment;
    var averageCaseObj = bestAndWorstCases.averageCaseObj;

    var worstCaseExperimentId = rowsOfWorstCaseExperiment.centroid[ColumnIndex.experimentId];
    var bestCaseExperimentId = rowsOfBestCaseExperiment.centroid[ColumnIndex.experimentId];
    Logger.log("worstCaseExperimentId: " + worstCaseExperimentId)
    Logger.log("bestCaseExperimentId: " + bestCaseExperimentId)
    if (!worstCaseExperimentId || !bestCaseExperimentId) { return false; }

    var worstCaseObj = getDataObjOfRowsTwin(rowsOfWorstCaseExperiment);
    var bestCaseObj = getDataObjOfRowsTwin(rowsOfBestCaseExperiment);    

    var today = new Date();
    var time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
    var dateTime = time;

    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(dateTime);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(filteredRows.length);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(filters.optim_algo);

    // worst case
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(worstCaseObj.centroidResult);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(worstCaseObj.optimizedResult);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(worstCaseObj.improvement);

    // average case
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(averageCaseObj.centroidResult);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(averageCaseObj.optimizedResult);
    getSheet(sheetName).getRange(cols[col_i++] + row_i).setValue(averageCaseObj.improvement);

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


function getDataObjOfRowsTwin(rowsObject) {
    var rowCentroid = rowsObject.centroid;
    var rowOptimized = rowsObject.optimized;
    var experimentId = rowCentroid[ColumnIndex.experimentId];
    var centroidResult = rowCentroid[ColumnIndex.objective_metric_result];
    var optimizedResult = rowOptimized[ColumnIndex.objective_metric_result];
    var improvement = centroidResult === 0 ? 0 : optimizedResult / centroidResult - 1;
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



var filterChoosen = function (filters) {
    return function (row) {
        return row[ColumnIndex.optim_algo] === filters.optim_algo
            && row[ColumnIndex.objective_metric] === filters.objective_metric
            && row[ColumnIndex.we_model] === filters.we_model
            && row[ColumnIndex.ground_truth_name] === filters.ground_truth_name
    }
}


function getRowByExperimentId(rows, experimentId) {
    var rowsExperiment = rows.filter(function (row) {
        return row[ColumnIndex.experimentId] === experimentId;
    })
    return { // rowsObject
        centroid: rowsExperiment.filter(function (row) {
            return row[ColumnIndex.INFO] === "CENTROID";
        })[0],
        optimized: rowsExperiment.filter(function (row) {
            return row[ColumnIndex.INFO] === "OPTIMIZED";
        })[0],
    }
}


function getBestAndWorstCaseFrom(filteredRows) {

    var worstCaseCentroidValue = 999;
    var worstCaseValue = 999;
    var bestCaseValue = -1;
    var rowsOfWorstCaseExperiment = null;
    var rowsOfBestCaseExperiment = null;
    var experimentIds = [];
    var averageCaseObj = {
        centroidResult: -1,
        optimizedResult: -1,
        improvement: -1,
    }
    var centroidList = []
    var optimizedList = []
    var improvementList = []

    for (var i = 0; i < filteredRows.length; i++) {
        var row = filteredRows[i];
        var experimentId = row[ColumnIndex.experimentId];
        if (experimentIds.indexOf(experimentId) === -1) {
            experimentIds.push(experimentId);
            var experimentRows = getRowByExperimentId(filteredRows, experimentId);

            // prendi l'id della riga e recupera anche la riga del CENTROID
            var rowCentroid = experimentRows.centroid;
            var rowOptimized = experimentRows.optimized;

            // leggi objective_metric_result OPTIMIZED e sottrai a objective_metric_result CENTROID
            var objectiveMetricResultCentroid = rowCentroid[ColumnIndex.objective_metric_result];
            var objectiveMetricResultOptimized = rowOptimized[ColumnIndex.objective_metric_result];
            var actualDifference = objectiveMetricResultOptimized - objectiveMetricResultCentroid;

            // memorizza quella con l'improvement minore
            if (actualDifference < worstCaseValue
                || objectiveMetricResultCentroid < worstCaseCentroidValue
            ) {
                rowsOfWorstCaseExperiment = getACopy(experimentRows);
                worstCaseValue = actualDifference;
                worstCaseCentroidValue = objectiveMetricResultCentroid;
            }

            // memorizza quella con l'improvement maggiore
            if (actualDifference >= bestCaseValue) {
                rowsOfBestCaseExperiment = getACopy(experimentRows);
                bestCaseValue = actualDifference;
            }

            centroidList.push(objectiveMetricResultCentroid)
            optimizedList.push(objectiveMetricResultOptimized)
            var improvement = objectiveMetricResultCentroid === 0 ? 0 : objectiveMetricResultOptimized / objectiveMetricResultCentroid - 1;
            improvementList.push(improvement)
        }
    }
    averageCaseObj.centroidResult = centroidList.reduce(function (prev, curr) {return prev + curr}) / centroidList.length;
    averageCaseObj.optimizedResult = optimizedList.reduce(function (prev, curr) {return prev + curr}) / optimizedList.length;
    averageCaseObj.improvement = improvementList.reduce(function (prev, curr) {return prev + curr}) / improvementList.length;
    return {
        rowsOfWorstCaseExperiment: rowsOfWorstCaseExperiment,
        rowsOfBestCaseExperiment: rowsOfBestCaseExperiment,
        averageCaseObj: averageCaseObj,
    };
}


function getACopy(par) {
    return JSON.parse(JSON.stringify(par))
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

var allRows = getSheet(MySettings.sheetResultsName).getRange(MySettings.resultsRange);
function getFilteredRowsByFilters(filters) {
    var filteredRows = allRows.getValues()
    filteredRows = filteredRows.filter(filterChoosen(filters))
    return filteredRows;
}



