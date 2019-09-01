Logger.clear()

var ColumnIndex = function () { }
ColumnIndex.INFO = 7; // OPTIMIZED, CENTROID
ColumnIndex.optim_algo = 8; // BFGS, nelder-mead, powell.....
ColumnIndex.objective_metric = 9; // AP@k
ColumnIndex.objective_metric_result = 10; // 0.348842 ...
ColumnIndex.optim_message = 23; // Success..
ColumnIndex.we_model = 36; //  models/GoogleNews-vectors-negative300.magnitude
ColumnIndex.ground_truth_name = 42; // usa_states
ColumnIndex.experimentId = 43;

var MySettings = function () { }
MySettings.sheetWorstAndBestCasesName = "Worst&BestCases";
MySettings.sheetResultsName = "Results";
MySettings.resultsRange = "A1:AS5200";

var sheetWorstAndBestCases = getSheet(MySettings.sheetWorstAndBestCasesName);
var sheetResults = getSheet(MySettings.sheetResultsName);

function searchWorstAndBestCases() {

    var objective_metric_list = ["AP@k"];
    var optim_algo_list = ['powell', 'nelder-mead', 'BFGS', 'Newton-CG', 'CG', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'COBYLA'];
    // var optim_algo_list = ['nelder-mead', 'BFGS', 'Newton-CG', 'CG', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg', 'COBYLA'];
    var we_model_list = ["models/GoogleNews-vectors-negative300.magnitude", "models/glove.840B.300d.magnitude"];
    var ground_truth_name_list = ["usa_states", "universe_solar_planets", "king_of_rome", "period_7_element"];

    var row_i = 3;
    var total = [objective_metric_list, optim_algo_list, we_model_list, ground_truth_name_list]
        .reduce(function (prev, curr) {
            return ((prev && prev.length) || prev) * curr.length
        })
    Logger.info("total: " + total);
    sheetWorstAndBestCases.getRange("A1").setValue("Wait.. -" + total);
    var allRows = sheetResults.getRange(MySettings.resultsRange);
    Logger.info("allRows: " + allRows.length);

    optim_algo_list.forEach(function (optim_algo) {
        var filters1 = {
            only_successful: true,
            optim_algo: optim_algo,
        }
        var preFilteredRows = getFilteredRowsByFilters(allRows, filters1);
        Logger.info("preFilteredRows: " + preFilteredRows.length);

        objective_metric_list.forEach(function (objective_metric) {
            we_model_list.forEach(function (we_model) {
                ground_truth_name_list.forEach(function (ground_truth_name) {
                    var filters2 = {
                        objective_metric: objective_metric,
                        optim_algo: optim_algo,
                        we_model: we_model,
                        ground_truth_name: ground_truth_name,
                    }
                    var rwawabc = addRowWithAverageWorstAndBestCase(preFilteredRows, sheetWorstAndBestCases, filters2, row_i);
                    if (rwawabc) { // if something has been added...
                        row_i = row_i + 1;
                    }
                    total = total - 1;
                    sheetWorstAndBestCases.getRange("A1").setValue("Wait.. -" + total + " [" + row_i + "]");
                    deleteRow(sheetWorstAndBestCases, row_i)
                });
            });
        });
    });
}

var cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V']
function addRowWithAverageWorstAndBestCase(rows, sheet, filters, row_i) {

    var col_i = 0;
    Logger.log("PRE filteredRows.length: " + rows.length)
    var filteredRows = getFilteredRowsByFilters(rows, filters);
    // Logger.log('filters: ' + JSON.stringify(filters))
    Logger.log("filteredRows.length: " + filteredRows.length)
    if (filteredRows.length <= 0) { return false; }

    var bestAndWorstCases = getBestAndWorstCaseFrom(filteredRows)
    var rowsOfWorstCaseExperiment = bestAndWorstCases.rowsOfWorstCaseExperiment;
    var rowsOfBestCaseExperiment = bestAndWorstCases.rowsOfBestCaseExperiment;
    var averageCaseObj = bestAndWorstCases.averageCaseObj;
    if (!rowsOfWorstCaseExperiment || !rowsOfBestCaseExperiment) { return false; }

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

    sheet.getRange(cols[col_i++] + row_i).setValue(dateTime);
    sheet.getRange(cols[col_i++] + row_i).setValue(filteredRows.length / 2);
    sheet.getRange(cols[col_i++] + row_i).setValue(filters.optim_algo);

    // worst case
    sheet.getRange(cols[col_i++] + row_i).setValue(worstCaseObj.centroidResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(worstCaseObj.optimizedResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(worstCaseObj.improvement);

    // average case
    sheet.getRange(cols[col_i++] + row_i).setValue(averageCaseObj.centroidResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(averageCaseObj.optimizedResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(averageCaseObj.improvementFromAVG);
    sheet.getRange(cols[col_i++] + row_i).setValue(averageCaseObj.improvementFromList);

    // best case
    sheet.getRange(cols[col_i++] + row_i).setValue(bestCaseObj.centroidResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(bestCaseObj.optimizedResult);
    sheet.getRange(cols[col_i++] + row_i).setValue(bestCaseObj.improvement);


    sheet.getRange(cols[col_i++] + row_i).setValue(filters.ground_truth_name);
    sheet.getRange(cols[col_i++] + row_i).setValue(filters.we_model);
    sheet.getRange(cols[col_i++] + row_i).setValue(filters.objective_metric);


    sheet.getRange(cols[col_i++] + row_i).setValue(worstCaseObj.experimentId);
    sheet.getRange(cols[col_i++] + row_i).setValue(bestCaseObj.experimentId);
    return true;

}

function getImprovement(centroidResult, optimizedResult) {
    return centroidResult === 0 ? 0 : (optimizedResult / centroidResult) - 1;
}

function getDataObjOfRowsTwin(rowsObject) {
    var rowCentroid = rowsObject.centroid;
    var rowOptimized = rowsObject.optimized;
    var experimentId = rowCentroid[ColumnIndex.experimentId];
    var centroidResult = rowCentroid[ColumnIndex.objective_metric_result];
    var optimizedResult = rowOptimized[ColumnIndex.objective_metric_result];
    return {
        centroidResult: centroidResult,
        optimizedResult: optimizedResult,
        improvement: getImprovement(centroidResult, optimizedResult),
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
        return (!filters.only_successful || (row[ColumnIndex.optim_message] === '' || row[ColumnIndex.optim_message] === 'Optimization terminated successfully.'))
            && (!filters.optim_algo || row[ColumnIndex.optim_algo] === filters.optim_algo)
            && (!filters.objective_metric || row[ColumnIndex.objective_metric] === filters.objective_metric)
            && (!filters.we_model || row[ColumnIndex.we_model] === filters.we_model)
            && (!filters.ground_truth_name || row[ColumnIndex.ground_truth_name] === filters.ground_truth_name)
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
        improvementFromList: -1,
        improvementFromAVG: -1,
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
            if (experimentRows.centroid && experimentRows.optimized) {
                // prendi l'id della riga e recupera anche la riga del CENTROID
                var rowCentroid = experimentRows.centroid;
                var rowOptimized = experimentRows.optimized;

                // leggi objective_metric_result OPTIMIZED e sottrai a objective_metric_result CENTROID
                var objectiveMetricResultCentroid = rowCentroid[ColumnIndex.objective_metric_result];
                var objectiveMetricResultOptimized = rowOptimized[ColumnIndex.objective_metric_result];
                var actualDifference = objectiveMetricResultOptimized - objectiveMetricResultCentroid;

                // memorizza quella con l'improvement minore
                if (!rowsOfWorstCaseExperiment
                    || actualDifference < worstCaseValue
                    || objectiveMetricResultCentroid < worstCaseCentroidValue
                ) {
                    rowsOfWorstCaseExperiment = getACopy(experimentRows);
                    worstCaseValue = actualDifference;
                    worstCaseCentroidValue = objectiveMetricResultCentroid;
                }

                // memorizza quella con l'improvement maggiore
                if (!rowsOfBestCaseExperiment
                    || actualDifference >= bestCaseValue) {
                    rowsOfBestCaseExperiment = getACopy(experimentRows);
                    bestCaseValue = actualDifference;
                }

                centroidList.push(objectiveMetricResultCentroid);
                optimizedList.push(objectiveMetricResultOptimized);
                improvementList.push(getImprovement(objectiveMetricResultCentroid, objectiveMetricResultOptimized));
            } else {
                Logger.log("Error experimentId doesn't exists: " + experimentId);
                Logger.log("experimentRows.centroid: " + experimentRows.centroid);
                Logger.log("experimentRows.optimized: " + experimentRows.optimized);
            }
        }
    }

    averageCaseObj.centroidResult = centroidList.length > 0 ? centroidList.reduce(function (prev, curr) { return prev + curr }) / centroidList.length : 0;
    averageCaseObj.optimizedResult = optimizedList.length > 0 ? optimizedList.reduce(function (prev, curr) { return prev + curr }) / optimizedList.length : 0;
    averageCaseObj.improvementFromList = improvementList.length > 0 ? improvementList.reduce(function (prev, curr) { return prev + curr }) / improvementList.length : 0;
    averageCaseObj.improvementFromAVG = getImprovement(averageCaseObj.centroidResult, averageCaseObj.optimizedResult);
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

function getFilteredRowsByFilters(rows, filters) {
    var filteredRows = Array.isArray(rows) ? rows : rows.getValues();
    filteredRows = filteredRows.filter(filterChoosen(filters));
    return filteredRows;
}

function deleteRow(sheet, row) {
    cols.forEach(function (col) {
        sheet.getRange(col + row).setValue('');
    })
}

